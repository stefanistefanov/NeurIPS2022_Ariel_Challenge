import os
from pathlib import Path
from typing import List

import h5py
import pandas as pd
from jsonargparse import CLI

from adc.dataclasses import Planet
from adc.utils import save_pickle


def add_forward_model_parameters(gt_dir: Path, planets: List[Planet]):
    fm_parameters_df = pd.read_csv(
        os.path.join(gt_dir, 'FM_Parameter_Table.csv'), index_col='planet_ID'
    )
    for planet in planets:
        planet.fm_parameters = fm_parameters_df.loc[
            planet.id, Planet.TARGET_NAMES
        ].values


def add_gt_trace_data(gt_dir: Path, planets: List[Planet]) -> List[Planet]:
    gt_trace = h5py.File(gt_dir.joinpath('Tracedata.hdf5'), 'r')
    quartiles_df = pd.read_csv(
        gt_dir.joinpath('QuartilesTable.csv'), index_col='planet_ID'
    )
    planets_with_trace_gt = []
    for planet in planets:
        planet_trace = gt_trace[f'Planet_{planet.id}']
        planet_trace_data = planet_trace['tracedata']
        if planet_trace_data.ndim > 0:
            assert planet_trace_data.ndim == 2
            planet.trace_data = planet_trace_data[:]
            planet.trace_weights = planet_trace['weights'][:]
            planet_quartiles = quartiles_df.loc[
                planet.id, Planet.QUARTILES_NAMES
            ]
            assert planet_quartiles.notnull().all()
            planet.quartiles = planet_quartiles.values
            planets_with_trace_gt.append(planet)
    gt_trace.close()
    return planets_with_trace_gt


def create_subset_planet_examples(data_subset_dir: Path) -> List[Planet]:
    spectral_data_path = data_subset_dir.joinpath('SpectralData.hdf5')
    spectral_data = h5py.File(spectral_data_path, 'r')

    aux_df = pd.read_csv(
        data_subset_dir.joinpath('AuxillaryTable.csv'), index_col='planet_ID'
    )
    planets = []
    for planet_id, aux_row in aux_df.iterrows():
        aux_features = aux_row[Planet.AUX_FEATURE_NAMES].values
        planet_spectral_data = spectral_data[f'Planet_{planet_id}']
        planet = Planet(
            id=planet_id,
            spectrum=planet_spectral_data['instrument_spectrum'][:],
            spectrum_noise=planet_spectral_data['instrument_noise'][:],
            aux_features=aux_features,
        )
        planets.append(planet)
    spectral_data.close()
    return planets


def save_wavelength_grid_and_width(data_subset_dir: Path, output_dir: Path):
    # instrument_wlgrid and instrument_width are the same for each planet
    # save them only once and not per planet
    spectral_data_path = data_subset_dir.joinpath('SpectralData.hdf5')
    spectral_data = h5py.File(spectral_data_path, 'r')
    first_planet_spectral_data = spectral_data['Planet_0']
    instrument_wlgrid = first_planet_spectral_data['instrument_wlgrid'][:]
    instrument_width = first_planet_spectral_data['instrument_width'][:]
    save_pickle(instrument_wlgrid, output_dir.joinpath('instrument_wlgrid.pkl'))
    save_pickle(instrument_width, output_dir.joinpath('instrument_width.pkl'))
    spectral_data.close()


def create_planet_examples(
    data_dir: str, output_dir: str,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    training_data_dir = data_dir.joinpath('TrainingData')
    save_wavelength_grid_and_width(training_data_dir, output_dir)

    train_planets = create_subset_planet_examples(training_data_dir)
    gt_dir = training_data_dir.joinpath('Ground Truth Package')
    add_forward_model_parameters(gt_dir, train_planets)
    # Save planets with forward model parameters. No trace data is saved in
    # these planet examples
    save_pickle(
        train_planets, output_dir.joinpath('train_planets_forward_model.pkl')
    )

    planets_with_trace_gt = add_gt_trace_data(gt_dir, train_planets)
    save_pickle(
        planets_with_trace_gt,
        output_dir.joinpath('train_planets_trace_gt.pkl'),
    )

    test_data_dir = data_dir.joinpath('TestData')
    test_planets = create_subset_planet_examples(test_data_dir)
    save_pickle(test_planets, output_dir.joinpath('test_planets.pkl'))

    final_test_data_dir = data_dir.joinpath('final_evaluation_set')
    final_test_planets = create_subset_planet_examples(final_test_data_dir)
    save_pickle(
        final_test_planets, output_dir.joinpath('final_test_planets.pkl')
    )


def main():
    CLI(create_planet_examples, as_positional=False)


if __name__ == '__main__':
    main()
