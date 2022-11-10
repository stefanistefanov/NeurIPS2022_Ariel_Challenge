import numpy as np
import pandas as pd

from adc.constants import NUM_TARGETS, NUM_QUARTILES
from ucl_baseline.metric_light_track import light_track_metric


def calculate_light_score(predictions_by_planet_id, planet_by_id):
    predictions_quartiles = []
    target_quartiles = []
    planet_scores = []
    for planet_id, planet_predictions in predictions_by_planet_id.items():
        planet_predictions_quartiles = np.quantile(
            planet_predictions, [0.16, 0.5, 0.84], axis=0
        )
        planet_target_quartiles = planet_by_id[planet_id].quartiles.reshape(
            NUM_QUARTILES, NUM_TARGETS
        )
        planet_score = light_track_metric(
            planet_target_quartiles, planet_predictions_quartiles
        )

        predictions_quartiles.append(planet_predictions_quartiles)
        target_quartiles.append(planet_target_quartiles)
        planet_scores.append(
            {
                'planet_id': planet_id,
                'light_score': planet_score,
                'num_predictions': len(planet_predictions),
            }
        )
    predictions_quartiles = np.stack(predictions_quartiles, axis=0)
    target_quartiles = np.stack(target_quartiles, axis=0)
    light_score = light_track_metric(target_quartiles, predictions_quartiles)
    planet_scores_df = pd.DataFrame(planet_scores)
    return light_score, planet_scores_df
