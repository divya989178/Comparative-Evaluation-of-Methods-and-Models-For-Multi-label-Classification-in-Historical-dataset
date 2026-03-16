import torch
from torch import nn

class LeastConfidence(Strategy):
    def __init__(
        self,
        pool_size,
        setting: str = 'precomputed',
        engine: str = 'qwen',
        reduction: str = 'mean'
    ):
        super().__init__(pool_size,setting, engine)
        assert reduction in ['mean', 'sum', 'max']
        self.reduction = reduction

    def query(self, args, k, pool_features, model):
        pool_indices = self._get_pool_indices()
        pool_features = pool_features.select(pool_indices.tolist())
        pool_logits = mlc_predict(args, model, pool_features)
        uncertainties = []
        for logit in pool_logits:
            prob = torch.sigmoid(logit)
            confidence = torch.abs(prob-0.5)
            if self.reduction == 'mean':
              uncertainties.append(confidence.mean())
            elif self.reduction == 'sum':
              uncertainties.append(confidence.sum())
            elif self.reduction == 'max':
                uncertainties.append(confidence.max())
        uncertainties = torch.stack(uncertainties)
        lab_indices = torch.topk(uncertainties, k=k, largest=False)[1]
        lab_indices = [pool_indices[i] for i in lab_indices]
        return lab_indices
