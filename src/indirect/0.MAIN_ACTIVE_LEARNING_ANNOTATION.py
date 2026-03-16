import os
import ujson as json
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

class Args():
  dataset=data
  save_path='./models'
  load_path=''
  model_name="google-bert/bert-base-multilingual-cased"#
  train_batch_size=16
  test_batch_size=16
  gradient_accumulation_steps=1
  learning_rate=2.6e-5
  adam_epsilon=1e-6 #for reweight
  max_grad_norm=1.0
  warmup_ratio=0.01
  num_train_epochs=10
  max_train_steps=10000
  early_stopping_patience=3
    # active learning related
  quadratic_selection=False
  default=500
  init_samples=50
  acquisition_samples=50
  acquisition_time=9
  strategy='confidence' #confidence KMeansSampling entropy
    # annotator related
  engine='qwen' #mistral gemini claude
  annotator_setting='precomputed' #'The setting to retrieve demo.'
    # automatic reweighting strategy
  reweight= False
    # misc
  device='cuda:0'
  seed=82 #52 42 62 72 82
  notes=''
  push_to_hub=True
  model_name_on_hub='model_tes'

def active_learning_loop(args):
    # get log
    log_dir='/content/logs'
    os.makedirs(log_dir,exist_ok=True)
    args.log_file = os.path.join(log_dir,f'{args.strategy}--{args.notes}.log')
    # get device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device
    # get task type
    task_type = 'mlc'
    # get config, tokenizer & data processor
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    cache_name = f'cache_{args.annotator_setting}_{args.engine}'
    #DATASET
    data_processor = Processor(dataset=args.dataset, tokenizer=tokenizer, cache_name=cache_name)
    config.id2label = data_processor.get_id2tag()
    config.label2id = data_processor.get_tag2id()
    config.num_labels = len(config.id2label)

    # add config from args
    config.model_name = args.model_name
    # GETS DATA FROM PROCESSOR EACH TRAIN DEMO AND TEST
    pool_features = data_processor.get_features(split='train')
    test_features = data_processor.get_features(split='val')
    test_set_features = data_processor.get_features(split='test')
    assert args.strategy in [ 'entropy', 'confidence', 'kmeans','random']

    reduction = 'max'  #mean max
    # reduction = 'sum'

    #USES CONFIG
    # Fixed: Changed elif to if to start the conditional block correctly
    if args.strategy == 'confidence':
        strategy = LeastConfidence(len(pool_features),args.annotator_setting,args.engine,reduction)
    else:
        raise ValueError('Unknown method.')
    # compute num of init samples
    if args.quadratic_selection:
        factor = int(args.budget / ((args.acquisition_time + 1) ** 2))
        n_init_samples = factor
    else:
        n_init_samples = args.init_samples


    indices = strategy.init_labeled_data(n_sample=n_init_samples)
    print(f"DEBUG: Initialized indices: {indices}")

    records = strategy.update(indices, pool_features)
    print(f"DEBUG: Records from annotation: {records}")
    print(f"DEBUG: Number of records: {len(records)}")

    if len(records) > 0:
        data_processor.update_cache(records)
        data_processor.reload()
        pool_features = data_processor.get_features(split='train')
        for idx in indices[:3]:
          sample = pool_features[int(idx)]
          print(f"DEBUG: After reload - Sample {idx}: id={sample['id']}, labels={sample['labels']}")
    active_learning_iterator = range(args.acquisition_time + 1)
    # begin active learning loop

    # get model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name,problem_type="multi_label_classification", num_labels=config.num_labels, id2label=config.id2label, label2id=config.label2id)
    model.to(device)
    train = train_mlc

    for i in active_learning_iterator:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name,problem_type="multi_label_classification", num_labels=config.num_labels, id2label=config.id2label, label2id=config.label2id)
        model.to(device)
        print('========== begin active learning loop {} =========='.format(i))
        ugly_log(args.log_file, '========== begin active learning loop {} =========='.format(i))
        train_features = strategy.get_labeled_data(pool_features)
        # Convert dataset slice to pandas
        train_df = train_features.to_pandas()

        # Convert label vectors to dataframe
        train_labels_df = pd.DataFrame(train_df["labels"].tolist(), columns=list(config.id2label.values()))
        label_counts_train = train_labels_df.sum(axis=0)
        prop_train = train_labels_df.mean(axis=0)
        verification_df = pd.DataFrame({
            "train_count": label_counts_train,
            "train_prop": prop_train
        })
        print("\nLabel distribution in training set:")
        print(verification_df)
        print(f'# of training data: {len(train_features)}')
        print(f"Number of rows: {len(train_features)}")
        for idx in range(min(5, len(train_features))):
          sample = train_features[idx]
          print(f"Sample {idx}: id={sample['id']}, labels={sample['labels']}")
          non_none = sum(1 for s in train_features if s['labels'] is not None)
          print(f"Samples with non-None labels: {non_none}")
        #train a model, based on the train_features
        model,test_results = train(args, train_features, test_features,test_set_features, model, config.id2label, tokenizer)

        # get new data
        if i == args.acquisition_time:
            continue
        pool_indices = strategy._get_pool_indices()
        if len(pool_indices) == 0:
          print(f'========== Pool exhausted! All {len(train_features)} samples labeled. Stopping. ==========')
          break
        print('========== acquiring new data ==========')
        k = args.acquisition_samples
        #USES QUERY
        indices = strategy.query(args, k, pool_features, model)

        #annotate these new texts
        records = strategy.update(indices, pool_features)

        if len(records) > 0:
            #input the new_annotations n in cache
            data_processor.update_cache(records)
active_learning_loop(args)
            #loading in the updated dataset
            data_processor.reload()
            #get the new pool_features
            pool_features = data_processor.get_features(split='train')
