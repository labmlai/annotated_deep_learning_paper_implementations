import torch.nn.functional as F
from torch import nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, epsilon= 0.5, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred, target):
        n = pred.size()[-1]
        log_pred = F.log_softmax(pred, dim=-1)
        loss = -log_pred.sum(dim=-1).mean()
        nll = F.nll_loss(log_pred, target, reduction=self.reduction)
        out = (1-self.epsilon)*nll + self.epsilon*(loss / n)
        return out
