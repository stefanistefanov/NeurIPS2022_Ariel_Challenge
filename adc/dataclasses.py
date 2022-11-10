from dataclasses import dataclass
from typing import List, Optional, ClassVar

import numpy as np


@dataclass
class Planet:

    TARGET_NAMES = [
        'planet_temp',
        'log_H2O',
        'log_CO2',
        'log_CH4',
        'log_CO',
        'log_NH3',
    ]

    AUX_FEATURE_NAMES: ClassVar[List[str]] = [
        'star_distance',
        'star_mass_kg',
        'star_radius_m',
        'star_temperature',
        'planet_mass_kg',
        'planet_orbital_period',
        'planet_distance',
        'planet_radius_m',
        'planet_surface_gravity',
    ]

    QUARTILES_NAMES: ClassVar[List[str]] = [
        'T_q1',
        'log_H2O_q1',
        'log_CO2_q1',
        'log_CH4_q1',
        'log_CO_q1',
        'log_NH3_q1',
        'T_q2',
        'log_H2O_q2',
        'log_CO2_q2',
        'log_CH4_q2',
        'log_CO_q2',
        'log_NH3_q2',
        'T_q3',
        'log_H2O_q3',
        'log_CO2_q3',
        'log_CH4_q3',
        'log_CO_q3',
        'log_NH3_q3',
    ]

    id: int
    spectrum: np.ndarray  # shape: (52,)
    spectrum_noise: np.ndarray  # shape: (52,)
    aux_features: np.ndarray  # auxiliary features shape: (9,)

    # optional GT attributes
    # num_trace_points varies by planet
    trace_data: Optional[np.ndarray] = None  # shape: (num_trace_points, 6)
    trace_weights: Optional[np.ndarray] = None  # shape: (num_trace_points,)
    quartiles: Optional[np.ndarray] = None  # shape: (18,)
    # forward model parameters
    fm_parameters: Optional[np.ndarray] = None  # shape: (6,)
