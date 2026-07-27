import warnings
warnings.filterwarnings("ignore", message="triton not found.*", module="torch.utils.flop_counter")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

import os
import random
from typing import Dict, List, Optional

from PIL import Image

import lightning as L
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, F1Score, Precision, Recall



class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)

class ConvNeXtV2Classifier(nn.Module):
    STAGE_DIMS = [96, 192, 384, 768]

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        use_mha: bool = True,
        mha_stage: int = 2,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        assert 0 <= mha_stage <= 3
        self.use_mha = use_mha
        self.mha_stage = mha_stage

        self.backbone = timm.create_model(
            'convnextv2_tiny.fcmae_ft_in22k_in1k',
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
            features_only=False,
        )

        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            x = self.backbone.stem(x)
            for i, stage in enumerate(self.backbone.stages):
                x = stage(x)
                if i == mha_stage:
                    self.mha_feat_dim = x.shape[1]
                    self.mha_spatial = x.shape[2] * x.shape[3]
            self.final_dim = x.shape[1]

        if use_mha:
            self.mha = nn.MultiheadAttention(
                embed_dim=self.mha_feat_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
            self.mha_norm = nn.LayerNorm(self.mha_feat_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(self.final_dim, num_classes)
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(alpha=1.0),
            nn.Linear(self.final_dim, 256),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 2)
        )

    def _apply_mha(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        attn_out, _ = self.mha(tokens, tokens, tokens, need_weights=False)
        tokens = self.mha_norm(tokens + attn_out)
        return tokens.transpose(1, 2).view(B, C, H, W)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.stem(x)
        for i, stage in enumerate(self.backbone.stages):
            x = stage(x)
            if self.use_mha and i == self.mha_stage:
                x = self._apply_mha(x)
        return x

    def forward(self, x: torch.Tensor, return_domain: bool = False) -> torch.Tensor:
        feat = self.forward_features(x)
        pooled = self.pool(feat).flatten(1)
        cls_logits = self.classifier(self.head_dropout(pooled))
        if return_domain:
            domain_logits = self.domain_classifier(pooled)
            return cls_logits, domain_logits
        return cls_logits


class GradCAMExtractor:
    def __init__(self, model: ConvNeXtV2Classifier, target_stage: int = 3):
        self.model = model
        self.target = model.backbone.stages[target_stage]
        self.gradients = None
        self.activations = None
        self._hooks = []

    def __enter__(self):
        def fwd_hook(m, i, o):
            self.activations = o

        def bwd_hook(m, gi, go):
            self.gradients = go[0]

        self._hooks.append(self.target.register_forward_hook(fwd_hook))
        self._hooks.append(self.target.register_full_backward_hook(bwd_hook))
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self.activations = None
        self.gradients = None

    @torch.enable_grad()
    def __call__(
        self,
        x: torch.Tensor,
        target_class: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if target_class is None:
            target_class = logits.argmax(dim=1)

        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
        logits.backward(gradient=one_hot, retain_graph=False)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)

        cam = cam.squeeze(1).detach().cpu().numpy()
        for i in range(cam.shape[0]):
            c_min, c_max = cam[i].min(), cam[i].max()
            cam[i] = (cam[i] - c_min) / (c_max - c_min) if c_max - c_min > 1e-8 else 0.0
        return cam.astype(np.float32)


def build_param_groups(
    model: ConvNeXtV2Classifier,
    base_lr: float,
    weight_decay: float,
    llrd: float = 0.75,
):
    groups = []

    head_params = list(model.classifier.parameters()) + list(model.head_dropout.parameters())
    if model.use_mha:
        head_params += list(model.mha.parameters()) + list(model.mha_norm.parameters())
    groups.append({'params': head_params, 'lr': base_lr, 'weight_decay': weight_decay})

    n_stages = len(model.backbone.stages)
    for i, stage in enumerate(model.backbone.stages):
        depth = n_stages - i
        lr = base_lr * (llrd ** depth)
        groups.append({'params': list(stage.parameters()), 'lr': lr, 'weight_decay': weight_decay})

    stem_lr = base_lr * (llrd ** (n_stages + 1))
    groups.append({'params': list(model.backbone.stem.parameters()), 'lr': stem_lr, 'weight_decay': weight_decay})

    return groups


class LeukemiaLightningModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        use_mha: bool = True,
        mha_stage: int = 2,
        num_heads: int = 8,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        llrd: float = 0.75,
        warmup_epochs: int = 5,
        max_epochs: int = 30,
        label_smoothing: float = 0.0,
        class_weights: Optional[List[float]] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 3.0,
        uda_mode: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ConvNeXtV2Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            use_mha=use_mha,
            mha_stage=mha_stage,
            num_heads=num_heads,
        )

        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer(
                'class_weight_tensor',
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weight_tensor = None

        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.val_prec = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.val_rec = Recall(task='multiclass', num_classes=num_classes, average='macro')

    BURNIN_EPOCHS = 20
    MIN_SOURCE_F1_FOR_PSEUDOLABEL = 0.70

    def on_train_epoch_start(self):
        if not self.hparams.uda_mode:
            return

        if self.current_epoch < self.BURNIN_EPOCHS:
            print(f"[UDA] Epoch {self.current_epoch}: burn-in ({self.current_epoch}/{self.BURNIN_EPOCHS}), skipping pseudo-label.")
            return

        source_f1 = self.trainer.callback_metrics.get('val_f1', torch.tensor(0.0))
        if isinstance(source_f1, torch.Tensor):
            source_f1 = source_f1.item()
        if source_f1 < self.MIN_SOURCE_F1_FOR_PSEUDOLABEL:
            print(f"[UDA] Epoch {self.current_epoch}: val_f1={source_f1:.4f} < {self.MIN_SOURCE_F1_FOR_PSEUDOLABEL}, skipping pseudo-label.")
            return

        dm = self.trainer.datamodule
        if not dm.target_samples:
            return

        epoch = self.current_epoch
        if epoch < 30:
            conf_threshold = 0.95
        elif epoch < 50:
            conf_threshold = 0.90
        else:
            conf_threshold = 0.85

        print(f"\n[UDA] Epoch {epoch}: Generating pseudo-labels "
              f"(threshold={conf_threshold}, source_f1={source_f1:.4f}, "
              f"{len(dm.target_samples)} target samples)...")
        self.model.eval()
        all_candidates: List[tuple] = []

        device = self.device
        transform = dm.val_transform

        class _TargetDS(Dataset):
            def __init__(self_, samples, transform):
                self_.samples = samples
                self_.transform = transform

            def __len__(self_):
                return len(self_.samples)

            def __getitem__(self_, idx):
                path, _ = self_.samples[idx]
                return self_.transform(Image.open(path).convert('RGB')), os.path.basename(path)

        loader = DataLoader(
            _TargetDS(dm.target_samples, transform),
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        with torch.inference_mode():
            for imgs, fnames in loader:
                imgs = imgs.to(device)
                logits = self(imgs)
                probs = torch.softmax(logits, dim=1)
                max_probs, pred_cls = torch.max(probs, dim=1)
                for fname, prob, cls in zip(fnames, max_probs.tolist(), pred_cls.tolist()):
                    if prob > conf_threshold:
                        all_candidates.append((fname, cls))

        class_buckets: Dict[int, List[str]] = {}
        for fname, cls in all_candidates:
            class_buckets.setdefault(cls, []).append(fname)

        pseudo_labels: Dict[str, int] = {}
        if class_buckets:
            if len(class_buckets) < 2:
                only_cls = list(class_buckets.keys())[0]
                print(f"[UDA] WARNING: All pseudo-labels are class {only_cls} only. "
                      f"Skipping injection to prevent bias reinforcement.")
                self.model.train()
                dm.update_pseudo_labels({})
                return
            min_count = min(len(v) for v in class_buckets.values())
            max_allowed = max(min_count * 4, 1)
            for cls, fnames in class_buckets.items():
                for fname in fnames[:max_allowed]:
                    pseudo_labels[fname] = cls
        else:
            max_allowed = 0

        self.model.train()
        print(f"[UDA] Kept {len(pseudo_labels)}/{len(dm.target_samples)} pseudo-labels "
              f"({len(all_candidates)} passed threshold, balanced cap={max_allowed}).")

        dm.update_pseudo_labels(pseudo_labels)

    def forward(self, x, return_domain=False):
        if return_domain:
            return self.model(x, return_domain=True)
        return self.model(x)

    def _focal_weights(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        return (1 - probs.gather(1, targets.unsqueeze(1)).squeeze(1)) ** self.hparams.focal_gamma

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict) and "source" in batch:
            source_batch = batch["source"]
            target_images = batch.get("target")
        else:
            source_batch = batch
            target_images = None
            
        images, targets_a, targets_b, lam = source_batch
        
        if target_images is not None:
            B_s = images.size(0)
            B_t = target_images.size(0)
            combined_images = torch.cat([images, target_images], dim=0)
            cls_logits, domain_logits = self(combined_images, return_domain=True)
            
            src_cls_logits = cls_logits[:B_s]
            
            domain_targets = torch.cat([
                torch.zeros(B_s, dtype=torch.long, device=self.device),
                torch.ones(B_t, dtype=torch.long, device=self.device)
            ])
            domain_loss = F.cross_entropy(domain_logits, domain_targets)
            
            logits = src_cls_logits
        else:
            logits = self(images)
            domain_loss = 0.0

        ce_a = F.cross_entropy(logits, targets_a, weight=self.class_weight_tensor,
                               reduction='none', label_smoothing=self.label_smoothing)
        ce_b = F.cross_entropy(logits, targets_b, weight=self.class_weight_tensor,
                               reduction='none', label_smoothing=self.label_smoothing)

        if self.hparams.use_focal_loss:
            fa = self._focal_weights(logits, targets_a)
            fb = self._focal_weights(logits, targets_b)
            cls_loss = (lam * fa * ce_a + (1 - lam) * fb * ce_b).mean()
        else:
            cls_loss = (lam * ce_a + (1 - lam) * ce_b).mean()

        total_loss = cls_loss + domain_loss
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        if target_images is not None:
            self.log('domain_loss', domain_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        if self.hparams.use_focal_loss:
            ce = F.cross_entropy(logits, labels, weight=self.class_weight_tensor,
                                 label_smoothing=self.label_smoothing, reduction='none')
            loss = (self._focal_weights(logits, labels) * ce).mean()
        else:
            loss = F.cross_entropy(logits, labels, weight=self.class_weight_tensor,
                                   label_smoothing=self.label_smoothing)

        preds = logits.argmax(dim=1)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_prec(preds, labels)
        self.val_rec(preds, labels)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_epoch=True)
        self.log('val_precision', self.val_prec, on_epoch=True)
        self.log('val_recall', self.val_rec, on_epoch=True)
        return loss

    def configure_optimizers(self):
        param_groups = build_param_groups(
            self.model, self.hparams.lr, self.hparams.weight_decay, self.hparams.llrd
        )
        optimizer = optim.AdamW(param_groups)

        warmup_steps = self.hparams.warmup_epochs
        total_steps = self.hparams.max_epochs

        def lr_lambda(epoch):
            if epoch < warmup_steps:
                return (epoch + 1) / warmup_steps
            progress = (epoch - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }
