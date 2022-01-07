import os
import numpy as np
from utils import load_json, get_word, recognize
import torch
from transformers import ElectraForSequenceClassification
from transformers import AutoTokenizer
import time
from audio_splitter_example import split_audio


label_dict=load_json(os.path.join('sample_data/label_dict.json'))
label_list=[x for x, y in label_dict.items()]

model_path=os.path.join('model')
model=ElectraForSequenceClassification.from_pretrained(model_path)
tokenizer=AutoTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')

