import os
import time
import ujson as json
from func_timeout.exceptions import FunctionTimedOut
from abc import ABC, abstractmethod
import numpy as np

RETRY = 3
class Strategy(ABC):
    def __init__(self, pool_size,annotator_type:str,train_knn_demo:str, setting: str='knn', engine: str,):
        self.lab_data_mask = np.zeros(pool_size, dtype=bool)
        self.annotator = Annotator(engine,annotator_type)
        self.setting = setting
        self.demo_file = {str(i['id']):i for i in demo}
        if setting == 'knn':
            demo_index_path = train_knn_demo
            self.demo_index = json.load(open(demo_index_path, 'r', encoding='utf-8'))
        elif setting == 'zero':
            pass
        else:
            raise ValueError(f'Unknown setting {setting}.')

    def __len__(self):
        return len(self.lab_data_mask)

    def _get_labeled_indices(self):
        return np.where(self.lab_data_mask)[0]

    def _get_pool_indices(self):
        return np.where(~self.lab_data_mask)[0]

    def get_labeled_data(self, features):
        labeled_indices = self._get_labeled_indices()
        labeled_data = features.select(labeled_indices)
        return labeled_data

    @abstractmethod
    def query(self, args, k, model, features):
        return

    def init_labeled_data(self, n_sample: int = None):
        if n_sample is None:
            raise ValueError('Please specify initial sample ratio/size.')
        if n_sample > len(self):
            raise ValueError('Initial sample size cannot be greater than the length of the data.')

        indices = np.arange(len(self))
        np.random.shuffle(indices)

        all_indices = indices[:n_sample]

        # Ensure both a and b are 1-dimensional arrays
        if all_indices.ndim != 1:
            raise ValueError("Mismatch in array dimensions or data types")


        # Update the labeled data mask
        self.lab_data_mask[all_indices] = True

        return all_indices

    def update(self, indices, features):
        self.lab_data_mask[indices] = True
        records = self.annotate(features)
        return records

    def annotate(self, features):
        results = {}
        labeled_indices = self._get_labeled_indices()
        for i in labeled_indices:
            feature = features[int(i)]
            label_key = 'labels'

            if feature[label_key] is None:
                if self.setting == 'knn':
                    demo = [self.demo_file[str(pointer['id'])] for pointer in reversed(self.demo_index[str(feature['id'])])]
                else:
                    demo = None

                result = None
                for j in range(RETRY):
                    try:
                        # Pass engine to online_annotate if annotator_type is 'close'
                        if self.annotator.annotator_type == 'close':
                            result = self.annotator.online_annotate(feature, self.annotator.annotator.engine, demo)
                        else:
                            result = self.annotator.online_annotate(feature, demo)
                        break
                    except FunctionTimedOut:
                        print('Timeout. Retrying')
                        time.sleep(60)

                if result is None:
                    print(f"Error: No annotation result for index {i} (feature id {feature['id']}).")
                else:
                  labels=[]
                  for i in result:
                    if isinstance(i,dict) and 'labels' in i:
                      labels.extend(i['labels'])
                  results[feature['id']] = [{"text":feature['text'],"labels":labels}]
        return results
