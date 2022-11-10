import os
from pathlib import Path

import numpy as np
import pandas as pd
from jsonargparse import CLI

from adc.constants import NUM_SUBMIT_SAMPLES
from adc.utils import load_pickle
from adc.utils import save_predictions


def create_ensemble_predictions(models_root_dir: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=False, parents=True)

    predictions_paths = [
        os.path.join(
            models_root_dir,
            str(model_idx),
            'predict',
            'final_test',
            'predictions.pkl.gz',
        )
        for model_idx in range(1, 11)
    ]

    lt_submission_paths = [
        os.path.join(
            models_root_dir,
            str(model_idx),
            'predict',
            'final_test',
            'lt_model_submission.csv',
        )
        for model_idx in range(1, 11)
    ]

    ensemble_predictions(output_dir, predictions_paths)

    output_lt_submission_path = os.path.join(
        output_dir, 'lt_model_submission.csv'
    )
    ensemble_lt_submissions(lt_submission_paths, output_lt_submission_path)


def ensemble_lt_submissions(lt_submission_paths, output_lt_submission_path):
    lt_submission_dfs = []
    for lt_submission_path in lt_submission_paths:
        lt_submission_df = pd.read_csv(
            lt_submission_path, index_col='planet_ID'
        )
        lt_submission_dfs.append(lt_submission_df)
    lt_submission_dfs = pd.concat(lt_submission_dfs)
    lt_ensemble_df = lt_submission_dfs.groupby(lt_submission_dfs.index).mean()
    lt_ensemble_df.to_csv(output_lt_submission_path)


def ensemble_predictions(output_dir, predictions_paths):
    predictions_list = [
        load_pickle(predictions_path, compress=True)['predictions']
        for predictions_path in predictions_paths
    ]
    planet_ids = None
    for predictions_path in predictions_paths:
        pred_dict = load_pickle(predictions_path, compress=True)
        pred_planet_ids = pred_dict['planet_id']
        if planet_ids is None:
            planet_ids = pred_planet_ids
        else:
            assert pred_planet_ids == planet_ids
        assert pred_dict['predictions'].shape[1] == NUM_SUBMIT_SAMPLES
        predictions_list.append(pred_dict['predictions'])

    num_model_samples = NUM_SUBMIT_SAMPLES // len(predictions_list)
    combined_predictions_by_planet_id = {}
    for planet_index, planet_id in enumerate(planet_ids):
        assert planet_index == planet_id
        planet_predictions = []
        for model_predictions in predictions_list:
            rng = np.random.default_rng(1337)
            trace_indices = rng.choice(
                NUM_SUBMIT_SAMPLES, num_model_samples, replace=False
            )
            planet_predictions.append(
                model_predictions[planet_index, trace_indices, :]
            )
        combined_predictions = np.concatenate(planet_predictions, axis=0)
        # remove one sample as the metric_regular_track.check_distribution is
        # strict < 5000.
        combined_predictions = combined_predictions[:-1, :]
        combined_predictions_by_planet_id[planet_id] = combined_predictions
    save_predictions(combined_predictions_by_planet_id, output_dir)


def main():
    CLI(create_ensemble_predictions, as_positional=False)


if __name__ == '__main__':
    main()
