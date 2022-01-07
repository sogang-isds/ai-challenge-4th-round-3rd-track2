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

def predict(input_file, tmp_path='./tmp'):
    split_audio(input_file, dest_path=tmp_path)

    files=os.listdir(tmp_path)
    files=sorted(files)

    wav_files=[]

    for file in files:
        wav_files.append(os.path.join(tmp_path, file))

    print('\nSpeech recognizing...')
    trans = []
    for wav_file in wav_files:
        if not wav_file.split('/')[-1].startswith('chunk'):
            continue

        result = recognize(wav_file)
        transcript = result['alternative'][0]['transcript'] if len(result) > 0 else ''

        trans.append(transcript)

        try:
            print(f'recognized :{wav_file}, {transcript}')
        except UnicodeEncodeError:
            print(f'recognition failed : {wav_file}')

        time.sleep(0.05)

    return trans