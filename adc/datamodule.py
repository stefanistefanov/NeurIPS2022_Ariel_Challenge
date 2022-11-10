import logging
from pathlib import Path
from typing import List

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from adc.dataclasses import Planet
from adc.dataset import PlanetDataset
from adc.preprocessing import PlanetPreprocessor
from adc.utils import load_pickle

log = logging.getLogger(__name__)


class ArielDataModule(LightningDataModule):
    def __init__(
        self,
        examples_data_dir: str,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        fold: int,
    ):
        super().__init__()
        self.examples_data_dir = Path(examples_data_dir)
        self.fold = fold
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def _fold_df_train_val_split(self):
        fold_df_path = self.examples_data_dir.joinpath('fold.csv')
        fold_df = pd.read_csv(fold_df_path)
        train_df = fold_df.loc[fold_df['fold'] != self.fold]
        val_df = fold_df.loc[fold_df['fold'] == self.fold]
        return train_df, val_df

    def setup(self, stage=None):
        train_df, val_df = self._fold_df_train_val_split()
        train_planet_ids = set(train_df['planet_id'])
        val_planet_ids = set(val_df['planet_id'])

        planets_trace_gt: List[Planet] = load_pickle(
            self.examples_data_dir.joinpath('train_planets_trace_gt.pkl')
        )
        train_planets = [
            planet
            for planet in planets_trace_gt
            if planet.id in train_planet_ids
        ]
        val_planets = [
            planet for planet in planets_trace_gt if planet.id in val_planet_ids
        ]

        planets_fm_gt: List[Planet] = load_pickle(
            self.examples_data_dir.joinpath('train_planets_forward_model.pkl')
        )
        planets_trace_gt_ids = {planet.id for planet in planets_trace_gt}
        train_planets_fm_gt_only = [
            planet
            for planet in planets_fm_gt
            if planet.id not in planets_trace_gt_ids
        ]

        del planets_trace_gt, planets_trace_gt_ids, planets_fm_gt

        test_planets: List[Planet] = load_pickle(
            self.examples_data_dir.joinpath('test_planets.pkl')
        )
        final_test_planets: List[Planet] = load_pickle(
            self.examples_data_dir.joinpath('final_test_planets.pkl')
        )

        self.mixture_params_by_planet_id = load_pickle(
            self.examples_data_dir.joinpath('mixture_params_by_planet_id.pkl')
        )

        self.preprocessor = PlanetPreprocessor()
        self.preprocessor.fit(
            train_planets
            + train_planets_fm_gt_only
            + val_planets
            + test_planets
            + final_test_planets
        )

        self.train_dataset = PlanetDataset(
            train_planets * 5
            + train_planets_fm_gt_only
            + val_planets * 5
            + test_planets * 10
            + final_test_planets * 10,
            self.preprocessor,
            augment=True,
            sample_target_trace=True,
            mixture_params_by_planet_id=self.mixture_params_by_planet_id,
        )
        log.info(f'train_dataset size={len(self.train_dataset)}')

        self.val_dataset = PlanetDataset(
            val_planets,
            self.preprocessor,
            augment=False,
            mixture_params_by_planet_id=self.mixture_params_by_planet_id,
        )
        log.info(f'val_dataset size={len(self.val_dataset)}')

        self.test_dataset = PlanetDataset(
            test_planets,
            self.preprocessor,
            augment=False,
            mixture_params_by_planet_id=self.mixture_params_by_planet_id,
        )
        log.info(f'test_dataset size={len(self.test_dataset)}')

        self.final_test_dataset = PlanetDataset(
            final_test_planets,
            self.preprocessor,
            augment=False,
            mixture_params_by_planet_id=self.mixture_params_by_planet_id,
        )
        log.info(f'final_test_dataset size={len(self.final_test_dataset)}')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return [
            DataLoader(
                self.test_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
            ),
            DataLoader(
                self.final_test_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
            ),
        ]
