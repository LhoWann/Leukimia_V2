import warnings
import logging
import gc
from typing import Optional

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

torch.set_float32_matmul_precision('medium')

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
        focal_gamma=3.0,
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
        focal_gamma=3.0,
        max_epochs=100,
        warmup_epochs=5,
        batch_size=32,
        use_dataset_weighted_sampling=True,
    ),
}


class GradCAMRefresher(L.Callback):
    """Refresh GradCAM saliency maps every N epochs using batched inference.

    Batched version: processes cfg.batch_size images per forward+backward pass
    instead of one at a time — eliminates the per-image Python loop bottleneck
    that was the dominant CPU cost on the previous implementation.
    """

    def __init__(self, refresh_every: int = 5, target_stage: int = 3,
                 batch_size: int = 32):
        super().__init__()
        self.refresh_every = refresh_every
        self.target_stage = target_stage
        self.batch_size = batch_size

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.refresh_every != 0:
            return
        dm = trainer.datamodule
        if dm.aug_mode != 'focusmix_cam':
            return

        print(f"\n[GradCAM] Regenerating maps at epoch {trainer.current_epoch}")
        pl_module.eval()
        cam_maps = {}
        device = pl_module.device

        from PIL import Image as _PIL
        from torch.utils.data import Dataset as _DS, DataLoader as _DL

        class _RawDataset(_DS):
            def __init__(self_, samples, transform):
                self_.samples = samples
                self_.transform = transform

            def __len__(self_):
                return len(self_.samples)

            def __getitem__(self_, idx):
                path, _ = self_.samples[idx]
                img = _PIL.open(path).convert('RGB')
                return self_.transform(img), idx

        raw_ds = _RawDataset(dm.train_dataset.samples, dm.val_transform)
        raw_dl = _DL(
            raw_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        with GradCAMExtractor(pl_module.model, target_stage=self.target_stage) as cam_extractor:
            for imgs, idxs in raw_dl:
                imgs = imgs.to(device)
                cams = cam_extractor(imgs)
                for i, idx in enumerate(idxs.tolist()):
                    cam_maps[idx] = cams[i]

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


def _release_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_experiment(
    cfg: ExperimentConfig,
    data_dir: str = 'dataset',
    seed: int = 42,
    ckpt_root: str = 'checkpoints',
    log_root: str = 'logs',
    run_name: str = '',
    num_workers: int = 2,
    accumulate_grad: int = 1,
    source_ckpt: Optional[str] = None,
) -> Optional[str]:
    set_seed(seed, strict=False)
    run_name = run_name or f'{cfg.name}_seed{seed}'

    datamodule = LeukemiaDataModule(
        data_dir=data_dir,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
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
    if source_ckpt:
        print(f"Source checkpoint : {source_ckpt}")
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

    if source_ckpt:
        src_path = Path(source_ckpt)
        if src_path.exists():
            print(f"\n[Transfer] Loading weights from: {src_path}")
            ckpt_data = torch.load(str(src_path), map_location='cpu', weights_only=False)
            state = ckpt_data.get('state_dict', ckpt_data)
            missing, unexpected = model.load_state_dict(state, strict=False)
            del ckpt_data, state
            _release_gpu()
            if missing:
                print(f"[Transfer] Missing keys  : {len(missing)}")
            if unexpected:
                print(f"[Transfer] Unexpected keys: {len(unexpected)}")
            print("[Transfer] Done.\n")
        else:
            print(f"[Transfer] WARNING: checkpoint not found: {src_path}")

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
        callbacks.append(GradCAMRefresher(
            refresh_every=5,
            target_stage=3,
            batch_size=cfg.batch_size,
        ))
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
        accumulate_grad_batches=accumulate_grad,
        reload_dataloaders_every_n_epochs=1 if cfg.uda_mode else 0,
    )

    trainer.fit(model, datamodule=datamodule)

    best_path = callbacks[0].best_model_path
    print(f"\nBest checkpoint: {best_path}")
    if best_path:
        trainer.validate(model, datamodule=datamodule, ckpt_path=best_path)

    return best_path or None


def run_dual_pipeline(
    data_dir: str,
    seeds: list,
    ckpt_root: str,
    log_root: str,
    num_workers: int,
    accumulate_grad: int,
):
    """Two-stage pipeline: source_only then uda_pseudolabel with weight transfer.

    Stage 1 trains on labeled source data only. The best checkpoint is then used
    to initialize Stage 2, so pseudo-label generation starts from a trained model
    rather than random weights. GPU memory is cleared between stages.
    """
    for seed in seeds:
        print(f"\n{'#' * 60}")
        print(f"# Dual Pipeline  —  Seed {seed}")
        print(f"{'#' * 60}")

        best_src = run_experiment(
            EXPERIMENTS['source_only'],
            data_dir=data_dir,
            seed=seed,
            ckpt_root=ckpt_root,
            log_root=log_root,
            num_workers=num_workers,
            accumulate_grad=accumulate_grad,
        )

        _release_gpu()

        run_experiment(
            EXPERIMENTS['uda_pseudolabel'],
            data_dir=data_dir,
            seed=seed,
            ckpt_root=ckpt_root,
            log_root=log_root,
            num_workers=num_workers,
            accumulate_grad=accumulate_grad,
            source_ckpt=best_src,
        )

        _release_gpu()

        print(f"\n[Evaluation] Running evaluation for seed {seed}...")
        subprocess.run([
            sys.executable, "src/evaluate.py", 
            "--exp", "source_only", "uda_pseudolabel", 
            "--seed", str(seed)
        ], check=True)
        print(f"[Evaluation] Done for seed {seed}.\n")


def main():
    parser = argparse.ArgumentParser(description='UDA Leukemia Training')
    parser.add_argument('--exp', type=str, default='source_only',
                        choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--data-dir', type=str, default='dataset')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='One or more seeds. Example: --seeds 42 123 456')
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments sequentially')
    parser.add_argument('--dual', action='store_true',
                        help='Run source_only then uda_pseudolabel with checkpoint handoff')
    parser.add_argument('--ckpt-root', type=str, default='checkpoints',
                        help='Root directory for checkpoints')
    parser.add_argument('--log-root', type=str, default='logs',
                        help='Root directory for CSV logs')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader worker processes')
    parser.add_argument('--accumulate-grad', type=int, default=1,
                        help='Gradient accumulation steps (e.g. 2 = effective batch x2)')
    args = parser.parse_args()

    if args.dual:
        run_dual_pipeline(
            data_dir=args.data_dir,
            seeds=args.seeds,
            ckpt_root=args.ckpt_root,
            log_root=args.log_root,
            num_workers=args.num_workers,
            accumulate_grad=args.accumulate_grad,
        )
    elif args.all:
        for name, cfg in EXPERIMENTS.items():
            for seed in args.seeds:
                run_experiment(
                    cfg, args.data_dir, seed,
                    ckpt_root=args.ckpt_root,
                    log_root=args.log_root,
                    num_workers=args.num_workers,
                    accumulate_grad=args.accumulate_grad,
                )
            _release_gpu()
    else:
        cfg = EXPERIMENTS[args.exp]
        for seed in args.seeds:
            run_experiment(
                cfg, args.data_dir, seed,
                ckpt_root=args.ckpt_root,
                log_root=args.log_root,
                num_workers=args.num_workers,
                accumulate_grad=args.accumulate_grad,
            )


if __name__ == '__main__':
    main()
