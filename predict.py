import os
import numpy as np
from utils import load_json, get_word
import torch
from transformers import ElectraForSequenceClassification
from transformers import AutoTokenizer

#label dict 불러오기
label_dict=load_json(os.path.join('sample_data/label_dict.json'))
label_list=[x for x, y in label_dict.items()]

#modle 및 tokenizer 불러오기
model_path=os.path.join('model')
model=ElectraForSequenceClassification.from_pretrained(model_path)
tokenizer=AutoTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')

#text 입력받고 추론
while True:
    try:
        text=get_word()

        if text=='exit':
            break

        model.eval()

        model_input=tokenizer.encode_plus(text)

        input={k:torch.tensor([v], dtype=torch.long) for k, v in model_input.items()} #모델에 들어갈 input
        input['labels']=None

        output=model(**input)
        logits=output[0]

        pred=logits.detach().cpu().numpy()
        pred_idx=np.argmax(pred, axis=1) #label dict와 모델의 클래스 인덱스 순서가 동일해야 한다.

        label=label_list[int(pred_idx)]
        print(f'{label}, {label_dict[label]}')

    except Exception as e:
        import traceback
        print(e)
        print(traceback.print_exc())