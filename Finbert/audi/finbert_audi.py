# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:53:57 2020

@author: victo
"""

#necessary libraries
import pandas as pd
import glob
import os

from pathlib import Path
import sys
sys.path.append('..')
import argparse
import shutil
import os
import logging
from textblob import TextBlob

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import *

from finbert.finbert import *
import finbert.utils as tools
from pprint import pprint
from sklearn.metrics import classification_report
#%load_ext autoreload
#%autoreload 2

project_dir = Path.cwd().parent
#project_dir = r'D:'
pd.set_option('max_colwidth', -1)
print(project_dir)

# file where csv files lies
path = r'C:\Users\victo\Master_Thesis\scraperproject\audi\audi_scraper\spiders\news'                     
all_files = glob.glob(os.path.join(path, "*.csv"))     

# read files to pandas frame
list_of_files = []

for filename in all_files:
    list_of_files.append(pd.read_csv(filename, 
                                     sep=',', 
                                     encoding='cp1252',
                                     header=None,
                                     names=["url", "header", "release time", "article content"])
                         )

# Concatenate all content of files into one DataFrame
concatenate_list_of_files = pd.concat(list_of_files, 
                                      ignore_index=True, 
                                      axis=0,
                                      )

# removing duplicates
cleaned_list_of_files = concatenate_list_of_files.drop_duplicates(keep=False, ignore_index=True)


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.ERROR)

lm_path = project_dir/'models'/'sentiment'/'TRC2'/'finbertTRC2'
cl_path = project_dir/'models'/'classifier_model'/'finbert-sentiment'
cl_data_path = r'C:/Users/victo/Master_Thesis/Finbert/data/sentiment_data/'

# Clean the cl_path
try:
    shutil.rmtree(cl_path) 
except:
    pass

bertmodel = BertForSequenceClassification.from_pretrained(r'D:/models/sentiment/FinancialPhraseBank/',cache_dir=None, num_labels=3)


config = Config(data_dir=cl_data_path,
                bert_model=bertmodel,
                num_train_epochs=4,
                model_dir=cl_path,
                max_seq_length = 48,
                train_batch_size = 32,
                learning_rate = 2e-5,
                output_mode='classification',
                warm_up_proportion=0.2,
                local_rank=-1,
                discriminate=True,
                gradual_unfreeze=True 
                )

finbert = FinBert(config)

finbert.prepare_model(label_list=['positive','negative','neutral'])

# Get the training examples
train_data = finbert.get_data('train')

model = finbert.create_the_model()

# This is for fine-tuning a subset of the model.

freeze = 11

for param in model.bert.embeddings.parameters():
    param.requires_grad = False
    
for i in range(freeze):
    for param in model.bert.encoder.layer[i].parameters():
        param.requires_grad = False     

#train the model
trained_model = finbert.train(train_examples = train_data, model = model)

#test the model
test_data = finbert.get_data('test')
results = finbert.evaluate(examples=test_data, model=trained_model)

#Prepare the classification report
def report(df, cols=['label','prediction','logits']):
    #print('Validation loss:{0:.2f}'.format(metrics['best_validation_loss']))
    cs = CrossEntropyLoss(weight=finbert.class_weights)
    loss = cs(torch.tensor(list(df[cols[2]])),torch.tensor(list(df[cols[0]])))
    print("Loss:{0:.2f}".format(loss))
    print("Accuracy:{0:.2f}".format((df[cols[0]] == df[cols[1]]).sum() / df.shape[0]) )
    print("\nClassification Report:")
    print(classification_report(df[cols[0]], df[cols[1]]))

results['prediction'] = results.predictions.apply(lambda x: np.argmax(x,axis=0))
report(results,cols=['labels','prediction','predictions'])

text = "Later that day Apple said it was revising down its earnings expectations in \
the fourth quarter of 2018, largely because of lower sales and signs of economic weakness in China. \
The news rapidly infected financial markets. Apple’s share price fell by around 7% in after-hours \
trading and the decline was extended to more than 10% when the market opened. The dollar fell \
by 3.7% against the yen in a matter of minutes after the announcement, before rapidly recovering \
some ground. Asian stockmarkets closed down on January 3rd and European ones opened lower. \
Yields on government bonds fell as investors fled to the traditional haven in a market storm."

cl_path = project_dir/'models'/'classifier_model'/'finbert-sentiment'
model = BertForSequenceClassification.from_pretrained(cl_path, cache_dir=None, num_labels=3)

result = predict(text,model)

blob = TextBlob(text)
result['textblob_prediction'] = [sentence.sentiment.polarity for sentence in blob.sentences]
result

print(f'Average sentiment is %.2f.' % (result.sentiment_score.mean()))

text2 = "Shares in the spin-off of South African e-commerce group Naspers surged more than 25% \
in the first minutes of their market debut in Amsterdam on Wednesday. Bob van Dijk, CEO of \
Naspers and Prosus Group poses at Amsterdam's stock exchange, as Prosus begins trading on the \
Euronext stock exchange in Amsterdam, Netherlands, September 11, 2019. REUTERS/Piroschka van de Wouw \
Prosus comprises Naspers’ global empire of consumer internet assets, with the jewel in the crown a \
31% stake in Chinese tech titan Tencent. There is 'way more demand than is even available, so that’s \
good,' said the CEO of Euronext Amsterdam, Maurice van Tilburg. 'It’s going to be an interesting \
hour of trade after opening this morning.' Euronext had given an indicative price of 58.70 euros \
per share for Prosus, implying a market value of 95.3 billion euros ($105 billion). The shares \
jumped to 76 euros on opening and were trading at 75 euros at 0719 GMT."

result2 = predict(text2,model)
blob = TextBlob(text2)
result2['textblob_prediction'] = [sentence.sentiment.polarity for sentence in blob.sentences]
result2



print(f'Average sentiment is %.2f.' % (result2.sentiment_score.mean()))






