import os
import re
import json
import pickle
import numpy as np
from tqdm import tqdm_notebook
from collections import Counter

import torch
from transformers import BertModel
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

base_path = 'LongSumm-data/extractive_summaries/'

path_to_jsons = base_path + 'papers-jsons/'

p_jsons = os.listdir(path_to_jsons)

p_jsons[:10]

p_unread = []

section_names = []
for p in p_jsons:
    with open(path_to_jsons+p) as json_file:
        try:
            p_data = json.load(json_file)
        except UnicodeDecodeError:
            p_unread.append(p)
            continue
    
    if p_data['metadata']['sections'] is not None:
        for s in p_data['metadata']['sections']:
            if s['heading'] is None:
                s['heading'] = 'abstract'
            s_name = re.sub(' +', ' ', re.sub('[^a-z\s]', '', s['heading'].lower())).lstrip()
            section_names.append(s_name)

unique_section_names = list(set(section_names))
unique_section_names

Counter(section_names).most_common(100)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

sents = p_jsons[:10]
sents

max_len = 5

section_names[:5]

input_ids = []
for s_name in tqdm_notebook(section_names):
    encoded_sent = tokenizer.encode(s_name, max_length = max_len, pad_to_max_length=True)
    input_ids.append(encoded_sent)

input_ids[:5]

attention_masks = []
for s_name_p in input_ids:
    att_mask = [int(token_id > 0) for token_id in s_name_p]
    attention_masks.append(att_mask)

attention_masks[:5]

input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)

batch_size = 128

data = TensorDataset(input_ids, attention_masks)
dataloader = DataLoader(data, batch_size = batch_size)

pooled_outs = []
for inputs, masks in tqdm_notebook(dataloader):
    outputs = model(input_ids = inputs, attention_mask = masks)
    pooled_outs += outputs[1].tolist()

len(section_names)

len(pooled_outs)

import pickle

# ls

with open('LongSumm-data/extractive_summaries/section_names_pooled_outs.pkl', 'wb') as f:
    pickle.dump(pooled_outs, f)

pooled_outs = np.array(pooled_outs)

model = AgglomerativeClustering(distance_threshold=None, n_clusters=10)

model = model.fit(pooled_outs)

labels = model.labels_

labels_l = labels.tolist()

len(labels_l)

with open('labels_list.pkl', 'wb') as f:
    pickle.dump(labels_l, f)
