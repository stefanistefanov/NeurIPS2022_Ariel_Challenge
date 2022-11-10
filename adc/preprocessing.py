import dataclasses
import logging
from typing import List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from adc.constants import MAX_SPECTRUM_VALUE
from adc.dataclasses import Planet
from ucl_baseline.metric_regular_track import default_prior_bounds

log = logging.getLogger(__name__)


class PlanetPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.aux_scaler = StandardScaler()
        self.target_bounds = default_prior_bounds().astype(np.float32)

    def fit(self, planets: List[Planet]):
        aux_features = np.stack([planet.aux_features for planet in planets])
        self.aux_scaler.fit(aux_features)

    def transform(self, planet: Planet) -> Planet:
        clipped_spectrum = np.clip(
            planet.spectrum, a_min=0.0, a_max=MAX_SPECTRUM_VALUE
        )

        scaled_aux_features = self.aux_scaler.transform(
            planet.aux_features[np.newaxis, :]
        ).squeeze(0)

        scaled_planet = dataclasses.replace(
            planet, spectrum=clipped_spectrum, aux_features=scaled_aux_features,
        )
        return scaled_planet

    def transform_targets(self, targets: np.ndarray):
        scaled_targets = (targets[np.newaxis, :] - self.target_bounds[:, 0]) / (
            self.target_bounds[:, 1] - self.target_bounds[:, 0]
        )
        scaled_targets = scaled_targets.squeeze(0)
        return scaled_targets

    def inverse_transform_targets(self, normalized_targets: np.ndarray):
        scale = self.target_bounds[:, 1] - self.target_bounds[:, 0]
        return self.target_bounds[:, 0] + scale * normalized_targets
