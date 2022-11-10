NUM_TARGETS = 6

NUM_QUARTILES = 3

NUM_SUBMIT_SAMPLES = 5000

# The number of predict samples is greater than number of submit samples because
# some predictions are rejected if they are not within default_prior_bounds
NUM_PREDICT_SAMPLES = 6000

# Extreme spectrum values are clipped to this value
MAX_SPECTRUM_VALUE = 0.5
