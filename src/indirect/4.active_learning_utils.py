from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from transformers import AutoModelForSequenceClassification

def mlc_collate_fn(batch):
    max_len =  max([len(f['input_ids']) for f in batch])
    input_ids = [f['input_ids'] + [0] * (max_len - len(f['input_ids'])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = [[1.0] * len(f['input_ids']) + [0.0] * (max_len - len(f['input_ids'])) for f in batch]
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    flag = True
    for f in batch:
        if f['labels'] is None:          # strategy ensures that training label is not None
            flag = False
            break
    if flag:
        labels = [f['labels'] + [0] * (max_len - len(f['labels'])) for f in batch]
        labels = torch.tensor(labels, dtype=torch.float)
    else:
        labels = None
    output = {'input_ids': input_ids, 'attention_mask': attention_mask,
              'labels': labels}
    return output

def mlc_predict(args, model, dataset): #for calculating uncertainity by any of the active learning technique
    model.eval()
    data_loader = DataLoader(dataset, batch_size=12, shuffle=False, collate_fn=mlc_collate_fn) #or: collate
    pred_logits = []
    max_len = 0
    for batch in tqdm(data_loader, desc='Evaluating on pool data'):
        with torch.no_grad():
            inputs = {'input_ids': batch['input_ids'].to(model.device),
                      'attention_mask': batch['attention_mask'].to(model.device)}
            outputs = model(**inputs)
            logits = outputs.logits
            pred_logits.append(logits)
    pred_logits = torch.cat(pred_logits,dim=0)
    return pred_logits


