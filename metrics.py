import torch
import torch.nn.functional as F
from torchmetrics import Metric



class BPC(Metric):
    def __init__(self, padding_idx=None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.padding_idx = padding_idx

        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_tokens", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: (B * L, V)
        targets: (B * L,)
        """

        #_, V = logits.shape

        #logits = logits.view(-1, V)
        #targets = targets.view(-1)

        if self.padding_idx is not None:
            mask = targets != self.padding_idx
            logits = logits[mask]
            targets = targets[mask]

        if targets.numel() == 0:
            return

        loss = F.cross_entropy(logits, targets, reduction="sum")

        self.total_loss += loss
        self.total_tokens += targets.numel()

    def compute(self):
        if self.total_tokens == 0:
            return torch.tensor(0.0)

        avg_loss_nat = self.total_loss / self.total_tokens
        bpc = avg_loss_nat / torch.log(torch.tensor(2.0))

        return bpc