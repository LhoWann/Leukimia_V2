import argparse
import os
import sys
from pathlib import Path

import optuna
import lightning as L
from optuna.integration import PyTorchLightningPruningCallback
from data_module import LeukemiaDataModule
from lightning_model import LeukemiaLightningModel

def objective(trial):
    base_lr = trial.suggest_float("base_lr", 1e-5, 1e-3, log=True)
    llrd = trial.suggest_float("llrd", 0.6, 0.95)
    focal_gamma = trial.suggest_float("focal_gamma", 1.0, 5.0)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    
    datamodule = LeukemiaDataModule(
        data_dir="dataset",
        batch_size=32,
        num_workers=2,
        aug_mode='focusmix',
        uda_mode=False
    )
    datamodule.setup('fit')
    
    model = LeukemiaLightningModel(
        num_classes=2,
        pretrained=True,
        lr=base_lr,
        weight_decay=weight_decay,
        llrd=llrd,
        max_epochs=15,
        use_focal_loss=True,
        focal_gamma=focal_gamma,
        uda_mode=False,
    )
    
    trainer = L.Trainer(
        max_epochs=15,
        accelerator='auto',
        devices='auto',
        precision='bf16-mixed',
        logger=False,
        enable_checkpointing=False,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_f1")],
    )
    
    trainer.fit(model, datamodule=datamodule)
    
    return trainer.callback_metrics.get("val_f1", 0.0).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=10)
    args = parser.parse_args()
    
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        study_name="leukemia_uda", 
        storage="sqlite:///optuna_study.db", 
        load_if_exists=True,
        direction="maximize", 
        pruner=pruner
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=14400) # 4 hours max
    
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == '__main__':
    main()
