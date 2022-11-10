from pytorch_lightning.cli import LightningCLI

from adc.datamodule import ArielDataModule
from adc.model import ArielNet
from adc.utils import config_logger


def main():
    config_logger()
    LightningCLI(ArielNet, ArielDataModule)


if __name__ == "__main__":
    main()
