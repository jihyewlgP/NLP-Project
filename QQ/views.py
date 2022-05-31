from django.shortcuts import render
from django.http import JsonResponse
import os

import json
import random
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel

from kobert import get_mxnet_kobert_model
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

from transformers import AutoTokenizer, AutoModel


# Create your views here.
from django.http import HttpResponse


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=4,   ##감정 수 조정 (분노, 불안, 기쁨, 슬픔) ##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


def index(request):
    return render(request, 'index.html')

def Data(request):
    # POST 요청일 때

    if request.method == 'POST':

        post = json.loads(request.body)
        print(post)
        if post is None:
            post = "다시 입력해주세요."



        if (post == '안녕'):
            row = {'reply': '안녕하세요.',
                    'emotion':'',
                    'singer':'',
                    'title':'',
                    'url':''
                   }
        elif (post == '안녕하세요'):
            row = {'reply': '안녕하세요.',
                    'emotion':'',
                    'singer':'',
                    'title':'',
                    'url':''
                   }           
        elif (post == '소개'):
            row = {'reply': '저는 QQ예요. 당신의 감정을 파악해 노래를 추천드리고 있어요.',
                    'emotion':'',
                    'singer':'',
                    'title':'',
                    'url':''
                   }
        else:
            
            extext = predict(post)

            ran = random.random() / 2 * 100

            ran = int(ran)

            print(ran)
            

            df = pd.read_excel('C:\Pegue\QQ_Project\Static\song_data.xlsx')
            
            emotion = 1            

            df.set_index("감정", inplace=True)      

            df2 = df.loc[[emotion]]

            

            row = {'reply':'지금 감정은 "' + extext + '"이네요',
                    'emotion':emotion,
                    'singer':df2.iloc[ran, 0],
                    'title':df2.iloc[ran, 1],
                    'url':df2.iloc[ran, 2]
                   }

        results = []       

        results.append(row)

        
    return HttpResponse(json.dumps(results))





def predict(predict_sentence):
    # Setting parameters
    max_len = 64
    batch_size = 64
    warmup_ratio = 0.1
    num_epochs = 5  
    max_grad_norm = 1
    log_interval = 200
    learning_rate =  5e-5

    bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")

    print('11')

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    
    # data_train = BERTDataset(dataset_train, 0, 1, tok, vocab, max_len, True, False)
    # data_test = BERTDataset(dataset_test,0, 1, tok, vocab,  max_len, True, False)
    
    device = torch.device("cuda:0")
    model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
    #model = torch.load('./kobert_model/model_state_dict.pt')
    model.load_state_dict(torch.load('./kobert_model/model_state_dict.pt'))


    print('22')


    

    data = [predict_sentence, '0']
    dataset_another = [data]

    print('33')

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    print('44')

    model.eval()

    print('55')

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)


        print('11111')


        valid_length= valid_length
        label = label.long().to(device)

        print('22222')

        out = model(token_ids, valid_length, segment_ids)

        print('33333')


        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("분노")
            elif np.argmax(logits) == 1:
                test_eval.append("불안")
            elif np.argmax(logits) == 2:
                test_eval.append("기쁨")
            elif np.argmax(logits) == 3:
                test_eval.append("슬픔")

        print('11111')

        print(">> 입력하신 감정은 " + test_eval[0] + " 입니다.") #감정에 따른 노래 추천 DB 연결 

        return test_eval[0]