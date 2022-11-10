import gzip
import logging
import os
import pickle
from typing import Dict

import numpy as np

from ucl_baseline.submit_format import (
    to_regular_track_format,
    to_light_track_format,
)

log = logging.getLogger(__name__)


def config_logger():
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger('adc')
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)


def save_pickle(obj, filepath, compress=False):
    with gzip.open(filepath, 'wb') if compress else open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath, compress=False):
    with gzip.open(filepath, 'rb') if compress else open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_predictions(
    predictions_by_planet_id: Dict[int, np.ndarray], output_dir: str
):
    prediction_planet_ids = sorted(predictions_by_planet_id.keys())
    predictions = []
    predictions_quartiles = []
    for planet_id in prediction_planet_ids:
        planet_predictions = predictions_by_planet_id[planet_id]
        planet_predictions_quartiles = np.quantile(
            planet_predictions, [0.16, 0.5, 0.84], axis=0
        )
        predictions.append(planet_predictions)
        predictions_quartiles.append(planet_predictions_quartiles)
    del predictions_by_planet_id
    predictions = np.stack(predictions, axis=0)
    predictions_quartiles = np.stack(predictions_quartiles, axis=0)
    weights = np.ones((predictions.shape[0], predictions.shape[1])) / np.sum(
        np.ones(predictions.shape[1])
    )
    to_regular_track_format(
        predictions,
        weights,
        name=os.path.join(output_dir, 'rt_submission.hdf5'),
    )
    to_light_track_format(
        predictions_quartiles[:, 0, :],
        predictions_quartiles[:, 1, :],
        predictions_quartiles[:, 2, :],
        name=os.path.join(output_dir, 'lt_submission.csv'),
    )
    save_pickle(
        {
            'predictions': predictions,
            'predictions_quartiles': predictions_quartiles,
            'planet_id': prediction_planet_ids,
        },
        os.path.join(output_dir, 'predictions.pkl.gz'),
        compress=True,
    )
