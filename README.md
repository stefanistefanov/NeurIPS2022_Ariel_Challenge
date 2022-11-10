The commands for solution generation are as follows.

`DATA_DIR` is the location of the competition data(TrainingData, TestData, 
final_evaluation_set directories inside) 

`EXAMPLES_DATA_DIR` is where planet examples and additional files
like mixture_params_by_planet_id.pkl, fold.csv are stored

Output artifacts will be stored in `OUTPUT_DIR`
```bash
DATA_DIR=<data_dir_path>
EXAMPLES_DATA_DIR=<examples_data_dir_path>
OUTPUT_DIR=<output_dir_path>
```

### Prepare planet examples
```bash
python -m adc.scripts.create_planet_examples \
  --data_dir $DATA_DIR \
  --output_dir $EXAMPLES_DATA_DIR
```

### Fit mixture models and save their parameters
```bash
python -m adc.scripts.create_mixture_models \
 --planet_examples_file $EXAMPLES_DATA_DIR/train_planets_trace_gt.pkl \
 --output_mixture_params_file $EXAMPLES_DATA_DIR/mixture_params_by_planet_id.pkl \
 --output_mixture_models_dir $OUTPUT_DIR/mixture_models
```

### Train
```bash
python -m adc.scripts.cli fit \
  --config config.yaml --seed_everything 1 \
  --data.examples_data_dir $EXAMPLES_DATA_DIR \
  --trainer.default_root_dir $OUTPUT_DIR/1/fit
```

### Predict
```bash
python -m adc.scripts.cli predict \
  --config config.yaml --seed_everything 1 \
  --data.examples_data_dir $EXAMPLES_DATA_DIR \
  --trainer.default_root_dir $OUTPUT_DIR/1/predict \
  --ckpt_path $OUTPUT_DIR/1/fit/lightning_logs/version_0/checkpoints/last.ckpt
```

### Ensemble
The training and predict commands above are executed 10 times changing 
`seed_everything` from 1 to 10 and `trainer.default_root_dir`/`ckpt_path`
accordingly.
The command for creating ensemble predictions is:
```bash
python -m adc.scripts.create_ensemble_predictions \
  --models_root_dir $OUTPUT_DIR --output_dir $OUTPUT_DIR/ensemble
```
