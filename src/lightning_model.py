import warnings
warnings.filterwarnings("ignore", message="triton not found.*", module="torch.utils.flop_counter")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

import random
from typing import Dict, List, Optional

import lightning as L
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy, F1Score, Precision, Recall


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        pooled = self.pool(feat).flatten(1)
        return self.classifier(self.head_dropout(pooled))




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
        focal_gamma: float = 2.0,
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
        self.register_buffer(
            'class_weight_tensor',
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None else None,
        )

        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.val_prec = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.val_rec = Recall(task='multiclass', num_classes=num_classes, average='macro')

    def on_train_epoch_start(self):
        # Iterative Pseudo-Labeling Phase
        if not self.hparams.uda_mode or self.current_epoch < 5:
            return

        dm = self.trainer.datamodule
        if not dm.target_samples:
            return
            
        print(f"\n[UDA] Epoch {self.current_epoch}: Generating pseudo-labels for {len(dm.target_samples)} target samples...")
        self.model.eval()
        pseudo_labels = {}
        all_candidates: List[tuple] = []

        device = self.device
        transform = dm.val_transform

        conf_threshold = 0.90

        with torch.inference_mode():
            for path, _ in dm.target_samples:
                from PIL import Image
                img_pil = Image.open(path).convert('RGB')
                x = transform(img_pil).unsqueeze(0).to(device)

                logits = self(x)
                probs = torch.softmax(logits, dim=1)[0]
                max_prob, pred_cls = torch.max(probs, dim=0)

                if max_prob.item() > conf_threshold:
                    from os.path import basename
                    all_candidates.append((basename(path), pred_cls.item()))

        class_buckets: Dict[int, List[str]] = {}
        for fname, cls in all_candidates:
            class_buckets.setdefault(cls, []).append(fname)

        if class_buckets:
            min_count = min(len(v) for v in class_buckets.values())
            max_allowed = max(min_count * 4, 1)
            for cls, fnames in class_buckets.items():
                for fname in fnames[:max_allowed]:
                    pseudo_labels[fname] = cls

        self.model.train()
        total_candidates = len(all_candidates)
        max_allowed = max(min(len(v) for v in class_buckets.values()) * 4, 1) if class_buckets else 0
        print(f"[UDA] Kept {len(pseudo_labels)}/{len(dm.target_samples)} pseudo-labels "
              f"(>{conf_threshold} threshold, {total_candidates} passed, balanced cap={max_allowed}).")

        dm.update_pseudo_labels(pseudo_labels)

    def forward(self, x):
        return self.model(x)

    def _focal_weights(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        return (1 - probs.gather(1, targets.unsqueeze(1)).squeeze(1)) ** self.hparams.focal_gamma

    def training_step(self, batch, batch_idx):
        images, targets_a, targets_b, lam = batch
        logits = self(images)

        ce_a = F.cross_entropy(logits, targets_a, weight=self.class_weight_tensor,
                               reduction='none', label_smoothing=self.label_smoothing)
        ce_b = F.cross_entropy(logits, targets_b, weight=self.class_weight_tensor,
                               reduction='none', label_smoothing=self.label_smoothing)

        if self.hparams.use_focal_loss:
            fa = self._focal_weights(logits, targets_a)
            fb = self._focal_weights(logits, targets_b)
            loss = (lam * fa * ce_a + (1 - lam) * fb * ce_b).mean()
        else:
            loss = (lam * ce_a + (1 - lam) * ce_b).mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

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
