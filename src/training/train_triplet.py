import torch
from torchvision import models
import pytorch_lightning as pl
import torch.nn.functional as F

# Triplet Loss
def triplet_loss(a, p, n, margin=0.2):
    d_ap = F.pairwise_distance(a, p)
    d_an = F.pairwise_distance(a, n)
    return F.relu(d_ap - d_an + margin).mean()

# Triplet Model
class TripletModel(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        base.fc = torch.nn.Linear(base.fc.in_features, 512)
        self.model = base
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        a, p, n = batch
        a_e, p_e, n_e = self(a), self(p), self(n)
        loss = triplet_loss(a_e, p_e, n_e)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
