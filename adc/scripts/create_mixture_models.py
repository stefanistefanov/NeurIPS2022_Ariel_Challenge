import math
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
from jsonargparse import CLI
from pomegranate.distributions import (
    NormalDistribution,
    UniformDistribution,
)
from pomegranate.gmm import GeneralMixtureModel
from tqdm import tqdm

from adc.dataclasses import Planet
from adc.preprocessing import PlanetPreprocessor
from adc.utils import save_pickle, load_pickle


def create_mixture_params_by_planet_id(
    planet_ids: List[int],
    mixture_results_dir: Path,
    output_mixture_params_file: Path,
):
    mixture_params_by_planet_id = {}
    for planet_id in planet_ids:
        mixture_results = load_pickle(
            mixture_results_dir.joinpath(f'{planet_id}.pkl')
        )
        mixture_params_by_planet_id[planet_id] = {
            'mean': np.array(mixture_results['normal_mean'], dtype=np.float32),
            'std': np.array(mixture_results['normal_std'], dtype=np.float32),
            'uniform_a': np.array(
                mixture_results['uniform_a'], dtype=np.float32
            ),
            'uniform_b': np.array(
                mixture_results['uniform_b'], dtype=np.float32
            ),
            'alpha': np.array(mixture_results['alpha'], dtype=np.float32),
        }
        for values in mixture_params_by_planet_id[planet_id].values():
            if np.isnan(values).any():
                print(f'nan values {planet_id} {values}')
    save_pickle(mixture_params_by_planet_id, output_mixture_params_file)


def create_planet_mixture_model(
    planet: Planet, output_dir: Path, normalize_trace_data=True
):
    planet_preprocessor = PlanetPreprocessor()
    trace_data = (
        planet_preprocessor.transform_targets(planet.trace_data)
        if normalize_trace_data
        else planet.trace_data
    )
    planet_targets_mean = (
        trace_data * planet.trace_weights[:, np.newaxis]
    ).sum(axis=0)
    planet_targets_cov = np.cov(
        trace_data, rowvar=0, aweights=planet.trace_weights
    )
    planet_targets_std = np.sqrt(np.diag(planet_targets_cov))
    independent_normal_distributions = [
        NormalDistribution(
            mean=planet_targets_mean[trace_index],
            std=planet_targets_std[trace_index],
        )
        for trace_index in range(trace_data.shape[1])
    ]

    uniform_a = planet_targets_mean - math.sqrt(3) * planet_targets_std
    uniform_b = planet_targets_mean + math.sqrt(3) * planet_targets_std
    independent_univariate_distributions = [
        UniformDistribution(uniform_a[trace_index], uniform_b[trace_index])
        for trace_index in range(trace_data.shape[1])
    ]
    alphas = [0.5] * 6
    mixture_models = []
    fit_normal_mean = []
    fit_normal_std = []
    fit_uniform_a = []
    fit_uniform_b = []
    fit_alpha = []
    for trace_index, (normal_distr, unif_distr, alpha) in enumerate(
        zip(
            independent_normal_distributions,
            independent_univariate_distributions,
            alphas,
        )
    ):
        mm = GeneralMixtureModel(
            [normal_distr, unif_distr], weights=[alpha, (1 - alpha)]
        )
        mm_before_fit = mm.copy()
        mm.fit(
            trace_data[:, trace_index],
            weights=planet.trace_weights,
            verbose=False,
            stop_threshold=1e-7,
            n_jobs=1,
        )
        # weights are in log space
        component_weights = np.exp(mm.weights[0])
        if (
            np.isnan(mm.distributions[0].parameters).any()
            or np.isnan(mm.distributions[1].parameters).any()
            or np.isnan(component_weights)
        ):
            print(f'planet_id={planet.id} nan in {trace_index}')
            mm = mm_before_fit

        mixture_models.append(mm)
        fit_normal_mean.append(mm.distributions[0].parameters[0])
        fit_normal_std.append(mm.distributions[0].parameters[1])
        fit_uniform_a.append(mm.distributions[1].parameters[0])
        fit_uniform_b.append(mm.distributions[1].parameters[1])
        fit_alpha.append(component_weights)

    mixture_result = {
        'normal_mean': fit_normal_mean,
        'normal_std': fit_normal_std,
        'uniform_a': fit_uniform_a,
        'uniform_b': fit_uniform_b,
        'alpha': fit_alpha,
        'mixture_models': mixture_models,
    }

    save_pickle(
        mixture_result, output_dir.joinpath(f'{planet.id}.pkl'),
    )


def create_mixture_models(
    planet_examples_file: str,
    output_mixture_models_dir: str,
    output_mixture_params_file: str,
):
    output_mixture_models_dir = Path(output_mixture_models_dir)
    output_mixture_models_dir.mkdir(exist_ok=False, parents=True)
    planets_trace_gt = load_pickle(planet_examples_file)
    create_planet_mixture_model_fn = partial(
        create_planet_mixture_model, output_dir=output_mixture_models_dir,
    )

    with mp.Pool(processes=4) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                create_planet_mixture_model_fn, planets_trace_gt
            ),
            desc='Fitting mixture models...',
            total=len(planets_trace_gt),
        ):
            pass

    planet_ids = [planet.id for planet in planets_trace_gt]
    create_mixture_params_by_planet_id(
        planet_ids,
        Path(output_mixture_models_dir),
        Path(output_mixture_params_file),
    )


def main():
    CLI(create_mixture_models, as_positional=False)


if __name__ == '__main__':
    main()
