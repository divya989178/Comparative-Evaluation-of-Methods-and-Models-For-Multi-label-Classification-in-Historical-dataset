import torch
from torch import nn
from kmeans_pytorch import kmeans

class entropy(Strategy):
    def __init__(self, pool_size, setting: str='', engine: str='',
                 reduction: str='mean',annotator_type:str='',train_knn_demo: str=''):
        super().__init__(pool_size,annotator_type,train_knn_demo, setting, engine)
        assert reduction in ['mean', 'sum', 'max']
        self.reduction = reduction

    def query(self, args, k, pool_features, model):
        pool_indices = self._get_pool_indices()
        pool_features = [pool_features[int(i)] for i in pool_indices]
        pool_logits = mlc_predict(args, model, pool_features)
        uncertainties = []
            for logit in pool_logits:
                probs = torch.sigmoid(torch.tensor(logit))
                entropy = torch.special.entr(probs)+torch.special.entr(1-probs)
                if self.reduction == 'mean':
                    uncertainties.append(entropy.mean())
                elif self.reduction == 'sum':
                    uncertainties.append(entropy.sum())
                elif self.reduction == 'max':
                    uncertainties.append(entropy.max())
            uncertainties = torch.stack(uncertainties)
        lab_indices = torch.topk(uncertainties, k=k)[1]
        lab_indices = [pool_indices[i] for i in lab_indices]
        return lab_indices

class KMeansSampling(Strategy):
    def __init__(self, pool_size, setting: str='', engine: str='',
                 annotator_type:str='',train_knn_demo: str=''):
        super().__init__(pool_size,annotator_type,train_knn_demo, setting, engine)

    def query(self, args, k, pool_features, model):
        pool_indices = self._get_pool_indices()
        pool_features = [pool_features[int(i)] for i in pool_indices]
        embeddings = get_bert_embeddings(args, pool_features, model)
            # compute k means centers
        ids, centers = kmeans(X=embeddings, num_clusters=k, device=args.device)
            # transfer back,since kmeans move data to cpu
        device = embeddings.device
        centers = centers.to(device)
        dist = torch.cdist(centers, embeddings)     # [n_clusters, n_samples]
        min_distances, lab_indices = torch.min(dist, dim=-1)
        indices =[pool_indices[i] for i in lab_indices]
        return indices

class LeastConfidence(Strategy):
    def __init__(self, pool_size, setting: str='', engine: str='',
                 reduction: str='mean',annotator_type:str='',train_knn_demo: str=''):
        super().__init__(pool_size,annotator_type,train_knn_demo, setting, engine)
        assert reduction in ['mean', 'sum', 'min']
        self.reduction = reduction

    def query(self, args, k, dataset, model, samples_likely_to_have_entity, k_enriched):
        pool_indices = self._get_pool_indices()
        pool_features = [dataset[int(i)] for i in pool_indices]
        pool_logits = ner_predict(args, model, pool_features)
        uncertainties = []
            for logit in pred_logits:
                prob = torch.sigmoid(logit)
                confidence = torch.abs(prob-0.5)*2
                confidence = 1 - confidence
                if self.reduction == 'mean':
                    uncertainties.append(confidence.mean())
                elif self.reduction == 'sum':
                    uncertainties.append(confidence.sum())
                elif self.reduction == 'min':
                    uncertainties.append(confidence.max())
            uncertainties = torch.stack(uncertainties)
        lab_indices = torch.topk(uncertainties, k=k)[1]
        lab_indices = [pool_indices[i] for i in lab_indices]
        return lab_indices
