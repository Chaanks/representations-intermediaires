import torch
from torch.utils import data
import numpy as np

import parser


class Dataset(data.Dataset):
    'Characterizes a dataset for Pytorch'
    def __init__(self, dataset_path):
        'Initialization'
        self.features, self.targets = parser.import_data(dataset_path)
        self.num_classes = 73
        self.labels = import_label('./classes')

    def __len__(self):
            'Denotes the total number of samples'
            return len(self.features)

    def __getitem__(self, idx):
            'Generates one sample of data'
            # Select sample
            x = self.features[idx]
            target = self.targets[idx]
            
            y = self.target_one_hot(target)
            
            return torch.from_numpy(x).float(), y


    def target_one_hot(self, str_code):
        idx = torch.tensor(np.where(self.labels == str_code)[0][0])
        #return torch.nn.functional.one_hot(idx, len(self.labels))
        return idx

    
def import_label(path):
    labels = []
    with open(path) as file:
        for line in file:
            if line.strip() == '':
                continue

            line = line.rstrip()
            labels.append(line)
    return np.asarray(labels)