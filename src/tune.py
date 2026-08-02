import argparse
import os
import sys
from pathlib import Path
import warnings
import logging

warnings.filterwarnings("ignore", message="triton not found.*")
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*Precision bf16-mixed is not supported.*")
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities._pytree").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.model_summary").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_WARNINGS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
os.chdir(ROOT)

import gc
import torch

import optuna
import lightning as L
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
from data_module import LeukemiaDataModule
from lightning_model import LeukemiaLightningModel


# NOTE: tune epochs = 25 so MS-DAST burn-in (epoch 20) is actually reached.
# Previous tuning used 15 epochs → pseudo-labeling never triggered → MS-DAST tuning was invalid.
TUNE_EPOCHS = 25


def objective(trial, ms_dast_mode=False):
    L.seed_everything(42, workers=True)

    # --- Search space (ConvNeXtV2-Tiny + MS-DAST-aware) ---
    base_lr      = trial.suggest_float("base_lr",      5e-6,  5e-5,  log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-3,  0.10,  log=True)
    llrd         = trial.suggest_float("llrd",         0.70,  0.90)
    focal_gamma  = trial.suggest_float("focal_gamma",  2.0,   5.0)
    label_smoothing  = trial.suggest_float("label_smoothing",  0.0,  0.15)
    warmup_epochs    = trial.suggest_int("warmup_epochs",      5,    15)
    domain_loss_weight = (
        trial.suggest_float("domain_loss_weight", 0.05, 0.5, log=True)
        if ms_dast_mode else 0.0
    )

    datamodule = LeukemiaDataModule(
        data_dir="dataset",
        batch_size=64,
        num_workers=4,
        aug_mode='focusmix',
        use_robust_aug=True,
        use_dataset_weighted_sampling=True,
        ms_dast_mode=ms_dast_mode,
    )
    datamodule.setup('fit')

    class_weights = datamodule.get_class_weights().tolist()

    model = LeukemiaLightningModel(
        num_classes=2,
        pretrained=True,
        lr=base_lr,
        weight_decay=weight_decay,
        llrd=llrd,
        max_epochs=TUNE_EPOCHS,
        warmup_epochs=warmup_epochs,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        use_focal_loss=True,
        focal_gamma=focal_gamma,
        ms_dast_mode=ms_dast_mode,
        domain_loss_weight=domain_loss_weight,
    )

    trainer = L.Trainer(
        max_epochs=TUNE_EPOCHS,
        accelerator='auto',
        devices=1,
        precision='16-mixed',
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=1 if ms_dast_mode else 0,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_f1")],
        gradient_clip_val=1.0,
    )

    try:
        trainer.fit(model, datamodule=datamodule)
        val_f1 = trainer.callback_metrics.get("val_f1", 0.0)
        return val_f1.item() if isinstance(val_f1, torch.Tensor) else float(val_f1)
    finally:
        del model, trainer, datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_study(n_trials, ms_dast_mode, timeout_sec):
    import time
    pruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=TUNE_EPOCHS, reduction_factor=3)
    mode_name = "ms_dast" if ms_dast_mode else "source"
    study_name = f"leukemia_{mode_name}_tiny"
    db_name    = f"optuna_{mode_name}_tiny.db"

    study = None
    for _attempt in range(5):
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=f"sqlite:///{db_name}?timeout=30",
                load_if_exists=True,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=pruner,
            )
            break
        except Exception as e:
            print(f"[run_study] create_study attempt {_attempt + 1}/5 failed: {e}. Retrying in 2s...")
            time.sleep(2)
    if study is None:
        raise RuntimeError(f"Failed to create/load Optuna study after 5 attempts: {db_name}")

    print(f"\n--- Starting tuning for {mode_name.upper()} on ConvNeXtV2-Tiny (n_trials={n_trials}, epochs={TUNE_EPOCHS}) ---")
    study.optimize(
        lambda trial: objective(trial, ms_dast_mode=ms_dast_mode),
        n_trials=n_trials,
        timeout=timeout_sec,
    )

    print(f"Finished trials: {len(study.trials)}")
    trial = study.best_trial
    print(f"Best val_f1: {trial.value:.4f}")
    print("Best params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='Optuna tuning for ConvNeXtV2-Tiny + MS-DAST leukemia')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials (default: 50)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Max seconds per study (default: None)')
    parser.add_argument('--ms-dast', action='store_true',
                        help='Tune MS-DAST mode. Run source tuning first.')
    parser.add_argument('--dual', action='store_true',
                        help='Run source then MS-DAST consecutively')
    args = parser.parse_args()

    if args.dual:
        run_study(args.n_trials, ms_dast_mode=False, timeout_sec=args.timeout)
        run_study(args.n_trials, ms_dast_mode=True,  timeout_sec=args.timeout)
    else:
        run_study(args.n_trials, ms_dast_mode=args.ms_dast, timeout_sec=args.timeout)


if __name__ == '__main__':
    main()