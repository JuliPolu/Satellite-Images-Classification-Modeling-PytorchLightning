import pytorch_lightning as pl
import torch
import torch.nn as nn
from timm import create_model
import numpy as np
from sklearn.metrics import f1_score

from src.config import Config
from src.losses import get_losses
from src.metrics import get_metrics
from src.utils import load_object


class PlanetModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config

        self._model = create_model(num_classes=self._config.num_classes, **self._config.model_kwargs)

        if self._config.freeze_grad:
            for param in self._model.parameters():
                param.requires_grad = False

            for param in list(self._model.parameters())[-(self._config.unfreeze_num):]:
                param.requires_grad = True

                #code for Resnet
                in_features = self._model.fc.in_features  
                self._model.fc = nn.Linear(in_features, self._config.num_classes) 

        self._losses = get_losses(self._config.losses)
        metrics = get_metrics(
            num_classes=self._config.num_classes,
            num_labels=self._config.num_classes,
            task='multilabel',
            average=self._config.metrics_average,
            threshold=self._config.threshold,
        )

        self.best_f1 = torch.zeros(self._config.num_classes)
        self.best_thresholds = torch.ones(self._config.num_classes) * 0.2  # Initialize to 0.2
        self.all_preds = []
        self.all_labels = []


        self._valid_metrics = metrics.clone(prefix='val_')
        self._test_metrics = metrics.clone(prefix='test_')

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def configure_optimizers(self):
        optimizer = load_object(self._config.optimizer)(
            self._model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx):
        """
        Считаем лосс.
        """
        images, gt_labels = batch
        pr_logits = self(images)
        return self._calculate_loss(pr_logits, gt_labels, 'train_')

    def validation_step(self, batch, batch_idx):
        """
        Считаем лосс и метрики.
        """
        images, gt_labels = batch
        pr_logits = self(images)
        
        pr_labels = torch.sigmoid(pr_logits)

        # Apply best_thresholds
        pr_labels = (pr_labels > self.best_thresholds.to(pr_labels.device)).float()
        
        self._calculate_loss(pr_logits, gt_labels, 'val_')
        self._valid_metrics(pr_labels, gt_labels)

        self.all_preds.append(pr_labels)
        self.all_labels.append(gt_labels)
        # return {'logits': pr_logits, 'labels': gt_labels}

    def test_step(self, batch, batch_idx):
        """
        Считаем метрики.
        """
        images, gt_labels = batch
        pr_logits = self(images)
        pr_labels = torch.sigmoid(pr_logits)

        # Apply best_thresholds
        pr_labels = (pr_labels > self.best_thresholds.to(pr_labels.device)).float()


        self._test_metrics(pr_labels, gt_labels)

    def on_validation_epoch_start(self) -> None:
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        val_metrics = self._valid_metrics.compute()
        self.log_dict(val_metrics, on_epoch=True)

        # Loop through each class to compute the best threshold
        # Concatenate all predictions and labels
        all_preds = torch.cat(self.all_preds, dim=0).cpu().numpy()
        all_labels = torch.cat(self.all_labels, dim=0).cpu().numpy()

        best_thresholds = []
        for i in range(all_preds.shape[1]):
            cur_best_thres = 0.2
            cur_best_score = 0
            for thres in np.arange(0.05, 0.5, 0.05):
                cur_score = f1_score(all_labels[:, i], all_preds[:, i] > thres)
                if cur_score > cur_best_score:
                    cur_best_score = cur_score
                    cur_best_thres = thres
            best_thresholds.append(cur_best_thres)

        self.best_thresholds = torch.tensor(best_thresholds)
        for i, t in enumerate(self.best_thresholds):
            self.log(f'best_thresholds_class_{i}', t.item(), on_epoch=True)

        # Clear all_preds and all_labels for the next epoch
        self.all_preds, self.all_labels = [], []


    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_metrics.compute(), on_epoch=True)

    def _calculate_loss(
        self,
        pr_logits: torch.Tensor,
        gt_labels: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        total_loss = 0
        for cur_loss in self._losses:
            loss = cur_loss.loss(pr_logits, gt_labels)
            total_loss += cur_loss.weight * loss
            self.log(f'{prefix}{cur_loss.name}_loss', loss.item())
        self.log(f'{prefix}total_loss', total_loss.item())
        return total_loss
