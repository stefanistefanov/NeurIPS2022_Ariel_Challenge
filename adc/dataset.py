import dataclasses
import logging
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from adc.constants import MAX_SPECTRUM_VALUE, NUM_QUARTILES, NUM_TARGETS
from adc.dataclasses import Planet
from adc.preprocessing import PlanetPreprocessor

log = logging.getLogger(__name__)


class PlanetDataset(Dataset):
    def __init__(
        self,
        planets: List[Planet],
        preprocessor: PlanetPreprocessor,
        augment: bool,
        sample_target_trace: bool = False,
        mixture_params_by_planet_id: Dict[int, Any] = None,
    ):
        super().__init__()
        self._planets = planets
        self.preprocessor = preprocessor
        self.augment = augment
        self.sample_target_trace = sample_target_trace
        self.mixture_params_by_planet_id = mixture_params_by_planet_id

    def source_planets(self):
        return self._planets

    def __getitem__(self, index):
        planet = self._planets[index]
        planet = self.preprocessor.transform(planet)

        # reconstruct original spectrum before augmentation
        target_spectrum = np.stack(
            (
                planet.spectrum,
                planet.spectrum_noise,
                planet.spectrum - planet.spectrum.mean(),
            ),
            axis=0,
        )

        if self.augment:
            if np.random.uniform() < 0.7:
                spectrum = planet.spectrum + np.random.normal(
                    loc=0, scale=planet.spectrum_noise,
                )
                # in case augmentation introduces extreme values
                spectrum = np.clip(
                    spectrum, a_min=0.0, a_max=MAX_SPECTRUM_VALUE
                )
                planet = dataclasses.replace(planet, spectrum=spectrum)

        spectrum_minus_mean = planet.spectrum - planet.spectrum.mean()

        spectrum_data = np.stack(
            (planet.spectrum, planet.spectrum_noise, spectrum_minus_mean),
            axis=0,
        )

        planet_dict = {
            'spectrum': torch.from_numpy(spectrum_data).float(),
            'aux_features': torch.from_numpy(planet.aux_features).float(),
            'planet_id': planet.id,
        }

        targets = self.get_planet_targets(planet)
        targets['target_spectrum'] = torch.from_numpy(target_spectrum).float()
        planet_dict['targets'] = targets

        return planet_dict

    def get_planet_targets(self, planet):
        targets = self.get_fm_parameters(planet)
        targets.update(self.get_trace_targets(planet))
        if 'trace_sample' not in targets:
            targets.update(
                {
                    'trace_sample': torch.zeros(NUM_TARGETS, dtype=torch.float),
                    'trace_sample_weight': torch.zeros(1, dtype=torch.float),
                }
            )
        return targets

    def get_fm_parameters(self, planet):
        result = {}
        if planet.fm_parameters is not None:
            normalized_fm_parameters = self.preprocessor.transform_targets(
                planet.fm_parameters
            )
            result.update(
                {
                    'fm_parameters': torch.from_numpy(
                        normalized_fm_parameters
                    ).float(),
                    'fm_parameters_weight': torch.ones(1, dtype=torch.float),
                }
            )

            if planet.trace_data is None:
                result.update(
                    {
                        'trace_sample': torch.from_numpy(
                            normalized_fm_parameters
                        ).float(),
                        'trace_sample_weight': torch.tensor([0.002]).float(),
                    }
                )
        else:
            result.update(
                {
                    'fm_parameters': torch.zeros(
                        NUM_TARGETS, dtype=torch.float
                    ),
                    'fm_parameters_weight': torch.zeros(1, dtype=torch.float),
                }
            )
        return result

    def get_trace_targets(self, planet):
        result = {}
        if planet.trace_data is not None:
            normalized_trace_data = self.preprocessor.transform_targets(
                planet.trace_data
            )
            planet_targets_mean = (
                normalized_trace_data * planet.trace_weights[:, np.newaxis]
            ).sum(axis=0)

            mixture_params = self.mixture_params_by_planet_id[planet.id]
            normalized_quartiles = self.preprocessor.transform_targets(
                planet.quartiles.reshape((NUM_QUARTILES, NUM_TARGETS))
            ).flatten()

            if self.sample_target_trace:
                trace_index = np.random.choice(
                    len(planet.trace_data), p=planet.trace_weights
                )
                trace_sample = normalized_trace_data[trace_index]
                trace_sample_weight = planet.trace_weights[
                    trace_index : trace_index + 1
                ]
            else:
                trace_sample_weight = np.max(
                    planet.trace_weights, keepdims=True
                )
                trace_sample = planet_targets_mean
            result.update(
                {
                    'mean': torch.from_numpy(mixture_params['mean']).float(),
                    'std': torch.from_numpy(mixture_params['std']).float(),
                    'uniform_a': torch.from_numpy(
                        mixture_params['uniform_a']
                    ).float(),
                    'uniform_b': torch.from_numpy(
                        mixture_params['uniform_b']
                    ).float(),
                    'alpha': torch.from_numpy(mixture_params['alpha']).float(),
                    'has_trace_gt': torch.ones(1, dtype=torch.float),
                    'quartiles': torch.from_numpy(normalized_quartiles).float(),
                    'trace_sample': torch.from_numpy(trace_sample).float(),
                    'trace_sample_weight': torch.from_numpy(
                        trace_sample_weight
                    ).float(),
                }
            )
        else:
            result.update(
                {
                    'mean': torch.zeros(NUM_TARGETS, dtype=torch.float),
                    'std': torch.zeros(NUM_TARGETS, dtype=torch.float),
                    'uniform_a': torch.zeros(NUM_TARGETS, dtype=torch.float),
                    'uniform_b': torch.zeros(NUM_TARGETS, dtype=torch.float),
                    'alpha': torch.zeros(NUM_TARGETS, dtype=torch.float),
                    'has_trace_gt': torch.zeros(1, dtype=torch.float),
                    'quartiles': torch.zeros(
                        NUM_QUARTILES * NUM_TARGETS, dtype=torch.float
                    ),
                }
            )

        return result

    def __len__(self):
        return len(self._planets)
