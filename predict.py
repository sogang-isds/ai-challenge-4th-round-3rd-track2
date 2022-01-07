import os
import numpy as np
from utils import load_json, get_word
import torch
from transformers import ElectraForSequenceClassification
from transformers import AutoTokenizer


label_dict=load_json(os.path.join('sample_data/label_dict.json'))
label_list=[x for x, y in label_dict.items()]

model_path=os.path.join('model')
model=ElectraForSequenceClassification.from_pretrained(model_path)
tokenizer=AutoTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')

while True:
    try:
        text=get_word()
        model.eval()

        model_input=tokenizer.encode_plus(text)

        input={k:torch.tensor([v], dtype=torch.long) for k, v in model_input.items()}
        input['labels']=None

        output=model(**input)
        logits=output[0]

        pred=logits.detach().cpu().numpy()
        pred_idx=np.argmax(pred, axis=1)

        label=label_list[int(pred_idx)]
        print(f'{label}, {label_dict[label]}')

    except Exception as e:
        import traceback
        print(e)
        print(traceback.print_exc())