import logging
import os
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from torch import nn, Tensor

from adc.metrics import calculate_light_score
from adc.constants import (
    NUM_SUBMIT_SAMPLES,
    NUM_PREDICT_SAMPLES,
    NUM_QUARTILES,
    NUM_TARGETS,
)
from adc.resnet1d import (
    BasicBlock,
    BasicDeconvBlock,
    ResNetBlock,
    ResNetDecoder,
)
from adc.utils import save_predictions
from passalis.dain import DAIN_Layer
from ucl_baseline.metric_light_track import light_track_metric
from ucl_baseline.submit_format import to_light_track_format

log = logging.getLogger(__name__)


class MixtureModelHeads(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.eps = torch.tensor(1e-6)
        self.linear_fm_parameters = nn.Linear(in_features, NUM_TARGETS)

        self.linear_mu = nn.Linear(in_features + NUM_TARGETS, NUM_TARGETS)

        self.linear_sigma = nn.Linear(in_features + NUM_TARGETS, NUM_TARGETS)

        self.linear_uniform_a = nn.Linear(
            in_features + NUM_TARGETS, NUM_TARGETS
        )

        self.linear_uniform_b = nn.Linear(
            in_features + NUM_TARGETS, NUM_TARGETS
        )

        self.linear_alpha = nn.Linear(in_features + NUM_TARGETS, NUM_TARGETS)

        self.linear_quartiles = nn.Linear(
            in_features + NUM_TARGETS, NUM_QUARTILES * NUM_TARGETS
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        fm_parameters = torch.sigmoid(self.linear_fm_parameters(x))
        x = torch.concat((x, fm_parameters), dim=1)

        mu = torch.sigmoid(self.linear_mu(x))
        sigma = F.softplus(self.linear_sigma(x))
        sigma = torch.maximum(sigma, self.eps)

        uniform_a = torch.sigmoid(self.linear_uniform_a(x))
        uniform_b = torch.sigmoid(self.linear_uniform_b(x))
        alpha = torch.sigmoid(self.linear_alpha(x))

        quartiles = torch.sigmoid(self.linear_quartiles(x))

        alpha = torch.maximum(alpha, self.eps)
        alpha = torch.minimum(alpha, 1.0 - self.eps)
        uniform_b = torch.maximum(uniform_b, uniform_a + self.eps)

        return mu, sigma, fm_parameters, uniform_a, uniform_b, alpha, quartiles


class MixtureModelEncoder(nn.Module):
    def __init__(self, in_channels, dropout):
        super().__init__()
        self.resnet_block = ResNetBlock(BasicBlock, in_channels=in_channels)

        dense_block_out_features = 256
        self.dense_block = nn.Sequential(
            nn.Linear(in_features=512 + 9, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=dense_block_out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.heads_block = MixtureModelHeads(dense_block_out_features)

    def forward(self, spectrum: Tensor, aux_features: Tensor) -> List[Tensor]:
        x = self.resnet_block(spectrum)
        batch_size = spectrum.shape[0]
        x = x.view(batch_size, -1)
        x = torch.concat((x, aux_features), dim=1)
        x = self.dense_block(x)
        heads_outputs = self.heads_block(x)
        return heads_outputs


class ArielNet(LightningModule):
    def __init__(
        self, dropout: float = 0.0, lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.eps = torch.tensor(1e-6)
        self.max_train_steps = None

        in_channels = 3
        self.dain = DAIN_Layer(mode='adaptive_scale', input_dim=in_channels)
        self.model = MixtureModelEncoder(in_channels, dropout)
        self.decoder = ResNetDecoder(
            BasicDeconvBlock, in_channels=36 + 18 + 9, out_channels=in_channels,
        )

    def on_fit_start(self) -> None:
        log.info(f'model={self}')
        self.max_train_steps = self.num_training_steps
        log.info(f'max_train_steps={self.max_train_steps}')

    def training_step(self, batch, batch_idx):
        spectrum = batch['spectrum']
        aux_features = batch['aux_features']
        targets = batch['targets']
        predictions = self(spectrum, aux_features)
        loss_dict = self.loss(predictions, targets)
        batch_size = len(targets)
        self._log_losses(
            loss_dict, batch_size, training=True,
        )
        return {'loss': loss_dict['loss']}

    @property
    def num_training_steps(self) -> int:
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.datamodule.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )
        effective_accum = self.trainer.accumulate_grad_batches
        return (batches // effective_accum) * self.trainer.max_epochs

    def _log_losses(self, loss_dict, batch_size, training, prefix=None):
        for loss_key, loss_value in loss_dict.items():
            self.log(
                f'{prefix}_{loss_key}' if prefix else loss_key,
                loss_value,
                on_step=training and loss_key == 'loss',
                on_epoch=True,
                prog_bar=loss_key == 'loss',
                batch_size=batch_size,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=2 * self.hparams.lr,
            pct_start=0.05,
            div_factor=30,
            final_div_factor=1e4,
            total_steps=self.num_training_steps,
            anneal_strategy='cos',
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }

    def forward(self, spectrum: Tensor, aux_features: Tensor):
        dain_spectrum = self.dain(spectrum)
        (
            mu,
            sigma,
            fm_parameters,
            uniform_a,
            uniform_b,
            alpha,
            quartiles,
        ) = self.model(dain_spectrum, aux_features)

        decoder_input = torch.concat(
            (
                mu,
                sigma,
                fm_parameters,
                uniform_a,
                uniform_b,
                alpha,
                quartiles,
                aux_features,
            ),
            dim=1,
        )

        reconstructed_spectrum = self.decoder(decoder_input)

        result = {
            'mu': mu,
            'sigma': sigma,
            'fm_parameters': fm_parameters,
            'uniform_a': uniform_a,
            'uniform_b': uniform_b,
            'alpha': alpha,
            'quartiles': quartiles,
            'reconstructed_spectrum': reconstructed_spectrum,
            'spectrum': spectrum,
        }
        return result

    def loss(self, model_output: Dict, targets: Dict):
        mixture_model_parameters_loss = self.loss_mixture_model_parameters(
            model_output, targets
        )
        nll_loss = self.loss_nll_mixture_model(model_output, targets)
        quartiles_loss = self.loss_quartiles(model_output, targets)
        fm_parameters_loss = self.loss_fm_parameters(model_output, targets)
        reconstruction_loss = self.loss_reconstruction(model_output, targets)

        mixture_model_parameters_weight = (
            (
                2.0
                - (2.0 - 0.01) * self.trainer.global_step / self.max_train_steps
            )
            if self.max_train_steps is not None
            else 1.0
        )

        loss_dict = {}
        loss_dict['loss'] = (
            mixture_model_parameters_weight * mixture_model_parameters_loss
            + fm_parameters_loss
            + quartiles_loss
            + 5 * nll_loss
            + reconstruction_loss
        )
        loss_dict[
            'mixture_model_parameters_loss'
        ] = mixture_model_parameters_loss.detach()
        loss_dict['nll_loss'] = nll_loss.detach()
        loss_dict['quartiles_loss'] = quartiles_loss.detach()
        loss_dict['fm_parameters_loss'] = fm_parameters_loss.detach()
        loss_dict['reconstruction_loss'] = reconstruction_loss.detach()
        return loss_dict

    def loss_nll_mixture_model(self, model_output: Dict, targets: Dict):
        normal_dist = torch.distributions.Normal(
            model_output['mu'], model_output['sigma'], validate_args=False,
        )

        uniform_dist = torch.distributions.Uniform(
            model_output['uniform_a'],
            model_output['uniform_b'],
            validate_args=False,
        )

        alpha = model_output['alpha']
        component_log_weights = torch.stack((alpha, 1 - alpha), dim=-1).log()
        component_log_probs = torch.stack(
            (
                normal_dist.log_prob(targets['trace_sample']),
                uniform_dist.log_prob(targets['trace_sample']),
            ),
            dim=-1,
        )
        nll_loss = -torch.logsumexp(
            component_log_probs + component_log_weights, dim=-1
        )
        nll_loss = targets['trace_sample_weight'] * nll_loss
        return nll_loss.mean()

    def loss_mixture_model_parameters(self, model_output: Dict, targets: Dict):
        mixture_model_parameters_loss = (
            F.l1_loss(model_output['mu'], targets['mean'], reduction='none')
            + F.l1_loss(model_output['sigma'], targets['std'], reduction='none')
            + F.l1_loss(
                model_output['uniform_a'],
                targets['uniform_a'],
                reduction='none',
            )
            + F.l1_loss(
                model_output['uniform_b'],
                targets['uniform_b'],
                reduction='none',
            )
            + F.l1_loss(
                model_output['alpha'], targets['alpha'], reduction='none'
            )
        )
        mixture_model_parameters_loss = (
            targets['has_trace_gt'] * mixture_model_parameters_loss
        )
        return mixture_model_parameters_loss.mean()

    def loss_quartiles(self, model_output: Dict, targets: Dict):
        quartiles_loss = F.l1_loss(
            model_output['quartiles'], targets['quartiles'], reduction='none'
        )
        quartiles_loss = targets['has_trace_gt'] * quartiles_loss
        return quartiles_loss.mean()

    def loss_fm_parameters(self, model_output, targets):
        fm_parameters_weight = targets['fm_parameters_weight']
        fm_parameters_loss = F.l1_loss(
            model_output['fm_parameters'],
            targets['fm_parameters'],
            reduction='none',
        )
        fm_parameters_loss = fm_parameters_weight * fm_parameters_loss
        fm_parameters_loss = fm_parameters_loss.mean()
        return fm_parameters_loss

    def loss_reconstruction(self, model_output: Dict, targets: Dict):
        l1_reconstruction_loss = F.l1_loss(
            model_output['reconstructed_spectrum'],
            targets['target_spectrum'],
            reduction='mean',
        )

        return l1_reconstruction_loss

    def validation_step(self, batch, batch_idx):
        val_dict = self.predict_step(batch, batch_idx)
        loss_dict = self.loss(val_dict['model_output'], batch['targets'])
        batch_size = len(batch['targets']['mean'])
        self._log_losses(loss_dict, batch_size, training=False, prefix='val')

        val_dict['model_output'] = {
            'mu': val_dict['model_output']['mu'].cpu().detach().numpy(),
            'sigma': val_dict['model_output']['sigma'].cpu().detach().numpy(),
            'quartiles': val_dict['model_output']['quartiles']
            .cpu()
            .detach()
            .numpy(),
        }

        return val_dict

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        model_output = self(batch['spectrum'], batch['aux_features'])
        sampled_predictions = self.sample_predictions_mixture_model_dist(
            model_output, NUM_PREDICT_SAMPLES
        )

        return {
            'planet_id': batch['planet_id'].cpu().numpy(),
            'predictions': sampled_predictions,
            'model_output': model_output,
        }

    def sample_predictions_mixture_model_dist(
        self, model_output_pass, num_samples
    ):
        # disable args validation for faster execution
        validate_args = False
        bernoulli_dist = torch.distributions.Bernoulli(
            model_output_pass['alpha'], validate_args=validate_args
        )
        normal_dist = torch.distributions.Normal(
            model_output_pass['mu'],
            model_output_pass['sigma'],
            validate_args=validate_args,
        )
        uniform_dist = torch.distributions.Uniform(
            model_output_pass['uniform_a'],
            model_output_pass['uniform_b'],
            validate_args=validate_args,
        )

        is_normal = bernoulli_dist.sample((num_samples,))
        normal_samples = normal_dist.sample((num_samples,))
        uniform_samples = uniform_dist.sample((num_samples,))
        batch_sampled_predictions = (
            is_normal * normal_samples + (1 - is_normal) * uniform_samples
        )
        batch_sampled_predictions = batch_sampled_predictions.transpose(0, 1)

        batch_sampled_predictions = (
            batch_sampled_predictions.cpu().detach().numpy().astype(np.float32)
        )
        return batch_sampled_predictions

    def epoch_predictions_by_planet_id(
        self, step_outputs
    ) -> Dict[int, List[np.ndarray]]:
        predictions_by_planet_id = {}
        for batch_output in step_outputs:
            for planet_id, planet_predictions in zip(
                batch_output['planet_id'], batch_output['predictions']
            ):
                predictions_by_planet_id[planet_id] = planet_predictions

        return predictions_by_planet_id

    def epoch_quartiles_by_planet_id(
        self, step_outputs
    ) -> Dict[int, np.ndarray]:
        preprocessor = self.trainer.datamodule.preprocessor
        quartiles_by_planet_id = {}
        for batch_output in step_outputs:
            for planet_id, planet_model_quartiles in zip(
                batch_output['planet_id'],
                batch_output['model_output']['quartiles'],
            ):
                if isinstance(planet_model_quartiles, torch.Tensor):
                    planet_model_quartiles = (
                        planet_model_quartiles.cpu().detach().numpy()
                    )
                planet_model_quartiles = preprocessor.inverse_transform_targets(
                    planet_model_quartiles.reshape(NUM_QUARTILES, NUM_TARGETS)
                )
                quartiles_by_planet_id[planet_id] = planet_model_quartiles

        return quartiles_by_planet_id

    def on_predict_epoch_end(self, results: List[Any]) -> None:
        super().on_predict_epoch_end(results)

        for result, subset in zip(results, ['test', 'final_test']):
            output_dir = os.path.join(self.trainer.default_root_dir, subset)
            os.makedirs(output_dir, exist_ok=True)

            (
                predictions_by_planet_id,
                predictions_stats_df,
            ) = self.epoch_outputs_to_predictions_by_planet(result)

            save_predictions(
                predictions_by_planet_id, output_dir,
            )

            self.save_lt_model_submission(result, subset)

            predictions_stats_df.to_csv(
                os.path.join(output_dir, f'predictions_stats.csv'), index=False,
            )

    def epoch_outputs_to_predictions_by_planet(self, outputs):
        predictions_by_planet_id = self.epoch_predictions_by_planet_id(outputs)
        preprocessor = self.trainer.datamodule.preprocessor
        rng = np.random.default_rng(int(os.environ.get('PL_GLOBAL_SEED')))
        target_limit_min = preprocessor.target_bounds[:, 0]
        target_limit_max = preprocessor.target_bounds[:, 1]
        predictions_stats = []
        for (
            planet_id,
            planet_predictions,
        ) in predictions_by_planet_id.items():
            planet_predictions_stats = {'planet_id': planet_id}
            planet_predictions = preprocessor.inverse_transform_targets(
                np.stack(planet_predictions, axis=0)
            )
            planet_predictions_stats['num_predictions_before_rejection'] = len(
                planet_predictions
            )
            planet_predictions_after_rejection = planet_predictions[
                (
                    (planet_predictions > target_limit_min)
                    & (planet_predictions < target_limit_max)
                ).all(axis=1),
                :,
            ]
            planet_predictions_stats['num_predictions_after_rejection'] = len(
                planet_predictions_after_rejection
            )

            if len(planet_predictions_after_rejection) == 0:
                log.warning(
                    f'No predictions within limits for planet_id={planet_id}.'
                    f' Using original predictions.'
                )
            else:
                planet_predictions = planet_predictions_after_rejection
                del planet_predictions_after_rejection

            if len(planet_predictions) > NUM_SUBMIT_SAMPLES:
                planet_predictions = rng.choice(
                    planet_predictions, size=NUM_SUBMIT_SAMPLES, replace=False
                )
            elif len(planet_predictions) < NUM_SUBMIT_SAMPLES:
                # resample predictions with replacement so that all have the
                # same size and to_regular_track_format function can be used
                planet_predictions = rng.choice(
                    planet_predictions, size=NUM_SUBMIT_SAMPLES, replace=True
                )
            planet_predictions_stats['num_predictions_final'] = len(
                planet_predictions
            )
            predictions_by_planet_id[planet_id] = planet_predictions
            predictions_stats.append(planet_predictions_stats)

        predictions_stats_df = pd.DataFrame(predictions_stats)
        return predictions_by_planet_id, predictions_stats_df

    def save_lt_model_submission(self, step_outputs, subset_dir=''):
        quartiles_by_planet_id = self.epoch_quartiles_by_planet_id(step_outputs)
        model_quartiles = []
        for (
            planet_id,
            planet_predictions_quartiles,
        ) in quartiles_by_planet_id.items():
            model_quartiles.append(planet_predictions_quartiles)
        model_quartiles = np.stack(model_quartiles, axis=0)
        to_light_track_format(
            model_quartiles[:, 0, :],
            model_quartiles[:, 1, :],
            model_quartiles[:, 2, :],
            name=os.path.join(
                self.trainer.default_root_dir,
                subset_dir,
                'lt_model_submission.csv',
            ),
        )

    def validation_epoch_end(self, outputs) -> None:
        (
            predictions_by_planet_id,
            _,
        ) = self.epoch_outputs_to_predictions_by_planet(outputs)
        planet_by_id = {
            planet.id: planet
            for planet in self.trainer.datamodule.val_dataset.source_planets()
        }
        val_light_score_epoch, planet_light_scores_df = calculate_light_score(
            predictions_by_planet_id, planet_by_id
        )
        self.log('val_light_score_epoch', val_light_score_epoch, prog_bar=True)
        log.info(
            f'epoch={self.current_epoch}'
            f' val_light_score_epoch={val_light_score_epoch:.2f}'
        )

        self.validation_model_light_scores(outputs)

    def validation_model_light_scores(self, outputs):
        quartiles_by_planet_id = self.epoch_quartiles_by_planet_id(outputs)
        planet_by_id = {
            planet.id: planet
            for planet in self.trainer.datamodule.val_dataset.source_planets()
        }
        model_quartiles = []
        target_quartiles = []
        model_light_scores = []
        for planet_id, planet_model_quartiles in quartiles_by_planet_id.items():
            planet_target_quartiles = planet_by_id[planet_id].quartiles.reshape(
                NUM_QUARTILES, NUM_TARGETS
            )
            planet_score = light_track_metric(
                planet_target_quartiles, planet_model_quartiles
            )
            model_quartiles.append(planet_model_quartiles)
            target_quartiles.append(planet_target_quartiles)
            model_light_scores.append(
                {'planet_id': planet_id, 'model_light_score': planet_score}
            )
        model_quartiles = np.stack(model_quartiles, axis=0)
        target_quartiles = np.stack(target_quartiles, axis=0)
        model_light_score = light_track_metric(
            target_quartiles, model_quartiles
        )
        self.log(
            'val_model_light_score_epoch', model_light_score, prog_bar=False
        )
        log.info(
            f'epoch={self.current_epoch}'
            f' val_model_light_score_epoch={model_light_score:.2f}'
        )
