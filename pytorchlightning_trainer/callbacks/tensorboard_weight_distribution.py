from typing import Optional, Tuple

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    import torchvision
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class TensorboardModelDistribution(Callback):
    """Generates the distribution of parameters or gradients and logs to tensorboard

    Example::
        from src.lightning.callbacks import TensorboardModelDistribution
        trainer = Trainer(callbacks=[TensorboardModelDistributio()])
    """

    def __init__(
        self,
        type="weight",
    ) -> None:
        """
        Args:

        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")
        super().__init__()

        if type not in ["weight", "gradient"]:
            raise NotImplementedError
        self.type = type

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        assert isinstance(trainer.logger.experiment, torch.utils.tensorboard.writer.SummaryWriter), \
            "The logger is not TensorBoard "
        if "weight" in self.type:
            for name, p in pl_module.model.named_parameters():
                if p.nelement() == 0:
                    continue
                if p.requires_grad:
                    trainer.logger.experiment.add_histogram("weight_" + name, p.data, trainer.global_step)
        else:
            raise NotImplementedError
