import torch
import higher
import evaluate
from functools
import numpy as np
from scipy.special import expit
from sklearn.metrics import f1_score,precision_score,recall_score,hamming_loss
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

def ugly_log(file, info):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(info + '\n')

def filter_none_labels(dataset):
    """
    Filters out examples with None labels from the dataset.
    """
    def has_valid_labels(example):
        # Check if the labels are not None
        return example['labels'] is not None

    # Filter the dataset
    return dataset.filter(has_valid_labels)

import numpy as np
from scipy.special import expit

from transformers import EvalPrediction
import torch

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    conf_mat,_ = mlcm.cm(y_true,y_pred,False)
    f = io.StringIO()
    with redirect_stdout(f):
       one_vs_rest = mlcm.stats(conf_mat)
    output = f.getvalue()
    weighted_f1 = float([l for l in output.split('\n') if 'weighted avg' in l][0].split()[4])
    metrics = {'f1': weighted_f1}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.labels.ids)
    return result

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.eval_dataset is not None:
            self.meta_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.data_collator
            )
            self.meta_iterator = cycle(self.meta_dataloader)
        if getattr(self.args, 'reweight', False):
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters()
                               if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters()
                               if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.meta_optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                eps=self.args.adam_epsilon if hasattr(self.args, 'adam_epsilon') else 1e-8,
            )

    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):

        if "labels" not in inputs or inputs["labels"] is None:
            ugly_log(self.args.log_file, "Labels are missing or None in compute_loss")
            if return_outputs:
                return torch.tensor(0.0), None
            return torch.tensor(0.0)
        # Check if reweighting is enabled and a meta optimizer is provided
        if getattr(self.args, 'reweight', False) and hasattr(self, 'meta_optimizer'):
            labels = inputs["labels"].to(self.args.device).float()  # Extract labels from inputs for custom handling
            model_inputs = {key: val.to(self.args.device) for key, val in inputs.items() if key!="labels"}  # Ensure all tensors are on the correct device
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')

            with higher.innerloop_ctx(model, self.meta_optimizer) as (fmodel, diffopt):
                # Forward pass on the pseudo model
                fmodel_outputs = fmodel(**model_inputs)
                logits = fmodel_outputs.logits
                per_label_loss = loss_fct(logits, labels)
                label_weights = self.label_weights.to(per_label_loss.device)
                per_sample_loss = (per_label_loss * label_weights).sum(dim=1) / label_weights.sum()

                batch_size = per_sample_loss.size(0)
                eps = torch.zeros(batch_size, requires_grad=True).to(self.args.device)
                meta_train_loss = torch.sum(eps * per_sample_loss)
                diffopt.step(meta_train_loss)

                # 2. Meta-validate on dev batch
                meta_batch = next(self.meta_iterator)
                meta_labels = meta_batch["labels"].to(self.args.device).float()
                meta_inputs = {k: v.to(self.args.device) for k, v in meta_batch.items() if k != "labels"}
                meta_outputs = fmodel(**meta_inputs)
                meta_val_loss = loss_fct(meta_outputs.logits, meta_labels).mean()

                eps_grads = torch.autograd.grad(meta_val_loss, eps)[0].detach()

            # 3. Normalize weights
            w_tilde = torch.clamp(-eps_grads, min=0)
            l1_norm = torch.sum(w_tilde)
            if l1_norm.item() > 0:
               w = w_tilde / l1_norm
            else:
                w = w_tilde

            # 4. Apply weights to real model
            real_outputs = model(**model_inputs)
            real_per_label_loss = loss_fct(real_outputs.logits, labels)
            real_per_sample_loss = (real_per_label_loss * label_weights).sum(dim=1) / label_weights.sum()
            reweighted_loss = torch.sum(w * real_per_sample_loss)

            if return_outputs:
                real_outputs.loss = reweighted_loss
                return reweighted_loss, real_outputs
            return reweighted_loss
        else:
            outputs = model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

from transformers import TrainingArguments, EarlyStoppingCallback
import io
from contextlib import redirect_stdout
def train_mlc(args, train_dataset, dev_dataset,test_dataset, model, id2label, tokenizer):
    train_dataset = filter_none_labels(train_dataset)
    dev_dataset = filter_none_labels(dev_dataset)

    max_index = max(id2label.keys())
    label_array = np.array([id2label[i] for i in range(max_index + 1)])

    data_collator = DataCollatorWithPadding(tokenizer) #look into this
    compute_metrics_with_id2label = partial(compute_metrics_with_report, label_array=label_array, log_file=args.log_file)

    training_args = TrainingArguments(
        output_dir=args.save_path,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=3,  # Keep only the most recent best model according to the evaluation
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        weight_decay=0.01,
        seed=args.seed,
        fp16=True,
        push_to_hub=True,
        hub_model_id=args.model_name_on_hub,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_with_id2label,
        callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        )
    ],
    )
    trainer.train()
    
    print("\n" + "="*80)
    print("FULL TEST CLASSIFICATION REPORT")
    print("="*80)

    predictions = trainer.predict(test_dataset)

    probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()
    pred_labels = (probs >= 0.5).astype(int)
    true_labels = predictions.label_ids

    conf_mat, _ = mlcm.cm(true_labels, pred_labels, False)

    f_output = io.StringIO()
    with redirect_stdout(f_output):
        mlcm.stats(conf_mat)

    formatted_output = f_output.getvalue()

    print(formatted_output)
    return trainer.model,test_results
