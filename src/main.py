import warnings
import logging

warnings.filterwarnings("ignore", message="triton not found.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities._pytree").setLevel(logging.ERROR)

import argparse
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

try:
    import numpy._core.multiarray
    import numpy.dtypes
    _safe = [numpy._core.multiarray.scalar, numpy.dtype]
    _safe += [getattr(numpy.dtypes, n) for n in dir(numpy.dtypes)
              if isinstance(getattr(numpy.dtypes, n), type)]
    torch.serialization.add_safe_globals(_safe)
except Exception:
    pass

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
)
from lightning.pytorch.loggers import CSVLogger
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO


class DirectDiskCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        torch.save(checkpoint, str(path))


from data_module import LeukemiaDataModule
from lightning_model import LeukemiaLightningModel, GradCAMExtractor

try:
    from lightning.pytorch.callbacks import RichProgressBar
    _RICH = True
except ImportError:
    _RICH = False


@dataclass
class ExperimentConfig:
    name: str
    aug_mode: str
    use_mha: bool
    uda_mode: bool = False
    mha_stage: int = 2
    aug_prob: float = 0.5
    paste_ratio: float = 0.25
    n_segments: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.05
    llrd: float = 0.75
    label_smoothing: float = 0.0
    batch_size: int = 32
    max_epochs: int = 100
    warmup_epochs: int = 5
    use_robust_aug: bool = True
    stain_sigma_mean: float = 0.15
    stain_sigma_std: float = 0.10
    stain_aug_prob: float = 0.5
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    use_dataset_weighted_sampling: bool = True


EXPERIMENTS = {
    'source_only': ExperimentConfig(
        name='source_only',
        aug_mode='focusmix',
        use_mha=False,
        uda_mode=False,
        paste_ratio=0.25,
        n_segments=50,
        use_robust_aug=True,
        stain_sigma_mean=0.15,
        stain_sigma_std=0.10,
        stain_aug_prob=0.5,
        max_epochs=100,
        warmup_epochs=5,
        batch_size=32,
        use_dataset_weighted_sampling=True,
    ),
    'uda_pseudolabel': ExperimentConfig(
        name='uda_pseudolabel',
        aug_mode='focusmix',
        use_mha=False,
        uda_mode=True,
        paste_ratio=0.25,
        n_segments=50,
        use_robust_aug=True,
        stain_sigma_mean=0.15,
        stain_sigma_std=0.10,
        stain_aug_prob=0.5,
        max_epochs=100,
        warmup_epochs=5,
        batch_size=32,
        use_dataset_weighted_sampling=True,
    ),
}


class GradCAMRefresher(L.Callback):
    def __init__(self, refresh_every: int = 5, target_stage: int = 3):
        super().__init__()
        self.refresh_every = refresh_every
        self.target_stage = target_stage

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.refresh_every != 0:
            return
        dm = trainer.datamodule
        if dm.aug_mode != 'focusmix_cam':
            return

        print(f"\n[GradCAM] Regenerating maps at epoch {trainer.current_epoch}")
        pl_module.eval()
        cam_maps = {}

        cam_transform = dm.val_transform
        device = pl_module.device

        with GradCAMExtractor(pl_module.model, target_stage=self.target_stage) as cam_extractor:
            for idx in range(len(dm.train_dataset)):
                img_pil, _ = dm.train_dataset.dataset[idx]
                x = cam_transform(img_pil).unsqueeze(0).to(device)
                cam = cam_extractor(x)
                cam_maps[idx] = cam[0]

        dm.train_dataset.set_gradcam_maps(cam_maps)
        pl_module.train()
        print(f"[GradCAM] Generated {len(cam_maps)} maps")


def set_seed(seed: int = 42, strict: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def run_experiment(cfg: ExperimentConfig, data_dir: str = 'dataset', seed: int = 42,
                   ckpt_root: str = 'checkpoints', log_root: str = 'logs',
                   run_name: str = ''):
    set_seed(seed, strict=False)
    run_name = run_name or f'{cfg.name}_seed{seed}'

    datamodule = LeukemiaDataModule(
        data_dir=data_dir,
        batch_size=cfg.batch_size,
        num_workers=2,
        aug_mode=cfg.aug_mode,
        aug_prob=cfg.aug_prob,
        n_segments=cfg.n_segments,
        paste_ratio=cfg.paste_ratio,
        use_robust_aug=cfg.use_robust_aug,
        stain_sigma_mean=cfg.stain_sigma_mean,
        stain_sigma_std=cfg.stain_sigma_std,
        stain_aug_prob=cfg.stain_aug_prob,
        use_dataset_weighted_sampling=cfg.use_dataset_weighted_sampling,
        uda_mode=cfg.uda_mode,
    )
    datamodule.setup()

    class_weights = datamodule.get_class_weights().tolist()

    print(f"\n{'=' * 60}")
    print(f"Experiment : {run_name}")
    print(f"UDA mode   : {cfg.uda_mode}")
    print(f"Seed       : {seed}")
    print(f"Config     : {asdict(cfg)}")
    print(f"Classes    : {datamodule.classes}")
    print(f"Train      : {len(datamodule.train_dataset)} | Val: {len(datamodule.val_dataset)}")
    print(f"Class weights: {[f'{w:.3f}' for w in class_weights]}")
    if cfg.uda_mode:
        print(f"Target (unlabeled): {len(datamodule.target_samples)} images")
    print(f"{'=' * 60}\n")

    model = LeukemiaLightningModel(
        num_classes=datamodule.num_classes,
        pretrained=True,
        use_mha=cfg.use_mha,
        mha_stage=cfg.mha_stage,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        llrd=cfg.llrd,
        warmup_epochs=cfg.warmup_epochs,
        max_epochs=cfg.max_epochs,
        label_smoothing=cfg.label_smoothing,
        class_weights=class_weights,
        use_focal_loss=cfg.use_focal_loss,
        focal_gamma=cfg.focal_gamma,
        uda_mode=cfg.uda_mode,
    )

    ckpt_dir = Path(ckpt_root) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='{epoch:02d}-{val_f1:.4f}',
            monitor='val_f1',
            mode='max',
            save_top_k=1,
            save_last=True,
            save_weights_only=True,
        ),
        EarlyStopping(monitor='val_f1', mode='max', patience=15, verbose=True),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    if cfg.aug_mode == 'focusmix_cam':
        callbacks.append(GradCAMRefresher(refresh_every=5))
    if _RICH:
        callbacks.append(RichProgressBar())

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator='auto',
        devices='auto',
        precision='bf16-mixed',
        callbacks=callbacks,
        logger=CSVLogger(log_root, name=run_name),
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        deterministic=False,
        reload_dataloaders_every_n_epochs=1 if cfg.uda_mode else 0,
        plugins=[DirectDiskCheckpointIO()],
    )

    trainer.fit(model, datamodule=datamodule)

    best_path = callbacks[0].best_model_path
    print(f"\nBest checkpoint: {best_path}")
    if best_path:
        trainer.validate(model, datamodule=datamodule, ckpt_path=best_path)

    return best_path


def main():
    parser = argparse.ArgumentParser(description='UDA Leukemia Training')
    parser.add_argument('--exp', type=str, default='source_only',
                        choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--data-dir', type=str, default='dataset')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='One or more seeds. Example: --seeds 42 123 456')
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments sequentially')
    args = parser.parse_args()

    if args.all:
        for name, cfg in EXPERIMENTS.items():
            for seed in args.seeds:
                run_experiment(cfg, args.data_dir, seed)
    else:
        cfg = EXPERIMENTS[args.exp]
        for seed in args.seeds:
            run_experiment(cfg, args.data_dir, seed)


if __name__ == '__main__':
    main()
