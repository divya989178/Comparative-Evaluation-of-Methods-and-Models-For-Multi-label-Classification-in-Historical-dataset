import os
import ujson as json
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

class Args():
  dataset=data
  save_path='./models'
  load_path=''
  model_name="emanjavacas/MacBERTh"
  train_batch_size=8
  test_batch_size=16
  gradient_accumulation_steps=1
  learning_rate=2e-5
  #adam_epsilon=1e-6 for reweight
  max_grad_norm=1.0
  warmup_ratio=0.06
  num_train_epochs=10
  max_train_steps=10000
  early_stopping_patience=5
    # active learning related
  default=2500
  init_samples=250
  acquisition_samples=250
  strategy='entropy' #confidence KMeansSampling
    # annotator related
  engine='qwen' #mistral gemini claude
  annotator_setting='knn' #'The setting to retrieve demo.'
  annotator_type="open" #close
    # automatic reweighting strategy
  reweight=False
    # misc
  device='cuda:0'
  seed=42
  notes=''
  push_to_hub=True,
  model_name_on_hub='model_t'

def active_learning_loop(args):
 
    log_dir='/content/logs'
    os.makedirs(log_dir,exist_ok=True)
    args.log_file = os.path.join(log_dir,f'{args.strategy}--{args.notes}.log')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    cache_name = f'cache_{args.annotator_setting}_{args.engine}' #stores labeled samples

  # processor creates the cache file (stores labeled samples) if it doesn't exist. Tokenizes the dataset.

    data_processor = Processor(dataset=args.dataset, tokenizer=tokenizer, cache_name=cache_name) 
    config.id2label = data_processor.get_id2tag()
    config.label2id = data_processor.get_tag2id()
    config.num_labels = len(config.id2label)

    config.model_name = args.model_name
   
    pool_features = data_processor.get_features(split='train')
    test_features = data_processor.get_features(split='val')
    assert args.strategy in [ 'entropy', 'confidence', 'kmeans']

    reduction = 'sum'  #mean max
  
    if args.strategy == 'entropy':
        strategy = EntropySampling(len(pool_features),args.annotator_setting,args.engine,args.annotator_type,'/content/train-knn-demo.json')  # sum for en_conll03
    elif args.strategy == 'confidence':
        strategy = LeastConfidence(len(pool_features),args.annotator_setting,args.engine,reduction,args.annotator_type,'/content/train-knn-demo.json')
    elif args.strategy == 'kmeans':
        strategy = KMeansSampling(len(pool_features),args.annotator_setting,args.engine,reduction,args.annotator_type,'/content/train-knn-demo.json')
    else:
        raise ValueError('Unknown method.')
    # compute num of init samples
   

    indices = strategy.init_labeled_data(n_sample=n_init_samples) #randomly selects initial samples
    records = strategy.update(indices, pool_features) # calls genAI to annotate the unlabeled samples  and returns labeled output

    if len(records) > 0:
        data_processor.update_cache(records) # updates cache file with annotated samples
        data_processor.reload() #reloads the training data with newly annotated samples
        pool_features = data_processor.get_features(split='train')
        for idx in indices[:3]:
          sample = pool_features[int(idx)] #pool_features==tokenized training data
    active_learning_iterator = range(args.acquisition_time + 1)
    # begin active learning loop

    # get model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name,problem_type="multi_label_classification", num_labels=config.num_labels, id2label=config.id2label, label2id=config.label2id)
    model.to(device)
    train = train_mlc

    for i in active_learning_iterator:
        print('========== begin active learning loop {} =========='.format(i))
        ugly_log(args.log_file, '========== begin active learning loop {} =========='.format(i))
        train_features = strategy.get_labeled_data(pool_features) #gets the labeled training data
        print(f'# of training data: {len(train_features)}')
        print(f"Number of rows: {len(train_features)}")
        for idx in range(min(5, len(train_features))):
          sample = train_features[idx]
          print(f"Sample {idx}: id={sample['id']}, labels={sample['labels']}")
          non_none = sum(1 for s in train_features if s['labels'] is not None)
          print(f"Samples with non-None labels: {non_none}")
      
        model = train(args, train_features, test_features, model, config.id2label, tokenizer)

        # get new data
        if i == args.acquisition_time:
            continue
        print('========== acquiring new data ==========')
        k = args.acquisition_samples
        #strategy.query gets unlabeled data, runs multi label predict function to get logits,
        #applies desired active learning technique to get samples the model is uncertain.
        indices = strategy.query(args, k, pool_features, model) 

        #gets the uncertain samples and sends it to genai annotator 
        records = strategy.update(indices, pool_features)

        if len(records) > 0:
            #put the new_annotations in cache
            data_processor.update_cache(records)
            #loading in the updated dataset
            data_processor.reload()
            #get the new pool_features
            pool_features = data_processor.get_features(split='train')
