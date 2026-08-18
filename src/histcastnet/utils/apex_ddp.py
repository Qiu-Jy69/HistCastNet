# Derived from Earthformer (Apache License 2.0).
# Modified for HistCastNet.
# Find the original code and discussion at https://github.com/PyTorchLightning/pytorch-lightning/discussions/10922
# We will need to use the AMP implementation from apex because https://discuss.pytorch.org/t/using-torch-utils-checkpoint-checkpoint-with-dataparallel/78452

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.overrides.base import (
    _LightningModuleWrapperBase,
    _LightningPrecisionModuleWrapperBase,
)


try:
    from apex.parallel import DistributedDataParallel as ApexDistributedDataParallel

    _HAS_APEX = True
except ImportError:
    ApexDistributedDataParallel = None
    _HAS_APEX = False


def unwrap_lightning_module(wrapped_model):
    model = wrapped_model

    if _HAS_APEX and isinstance(model, ApexDistributedDataParallel):
        model = unwrap_lightning_module(model.module)
    if isinstance(
        model, (_LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase)
    ):
        model = unwrap_lightning_module(model.module)
    return model


class ApexDDPStrategy(DDPStrategy):

    def __init__(self, *args, delay_allreduce=None, **kwargs):

        super().__init__(*args, **kwargs)

        if hasattr(self, "_ddp_kwargs"):
            self._ddp_kwargs.pop("delay_allreduce", None)

    def _setup_model(self, model):

        if _HAS_APEX and ApexDistributedDataParallel is not None:
            return ApexDistributedDataParallel(model)

        return super()._setup_model(model)

    @property
    def lightning_module(self):

        if _HAS_APEX and isinstance(self._model, ApexDistributedDataParallel):
            return unwrap_lightning_module(self._model)

        return super().lightning_module


if __name__ == "__main__":
    # Correct usage of apex DDP, which can avoid error caused by using `torch.utils.checkpoint`
    # when using `strategy="ddp"` in pl.
    import pytorch_lightning as pl

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=4,
        strategy=ApexDDPStrategy(find_unused_parameters=False),
    )
