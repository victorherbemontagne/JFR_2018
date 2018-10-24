from eventregistry import *

import numpy as np
import pandas as pd

import os
from tqdm import tqdm

os.chdir(r"D:\Deepnews\deepnews_github\Deepnews.ia\product\automatic extraction")
def load_json(path):
    import json
    f = open(path)
    data = json.load(f)
    return(data)

config = load_json("config.json.txt")
data = load_json('results_brut_event_registry.json')
## TAKES AT INPUT A JSON OF BRUT OUTPUT OF EVENT REGISTER
## THIS SCRIPT'S GOAL IS TO ORGANISE ARTICLE BY SUBJECTS, RUN THE SCORING ON THEM AND 
## MAKE THE FORMAT SUITABLE TO PUT IT IN THE INFORMATION MAXIMISATION TOOL


df_data = pd.DataFrame(data)
print("Data loaded in a DataFrame")

##Part where you load the model

#df_data = model.predict_df(df_data,with_topics=False)#? ça correspond à quoi with_topics? je vois pas trop ce que tu calcules
print("Model loaded ")

##Part where I sort articles by event

all_events = list(df_data['eventUri'].value_counts().keys())

df_data_events = df_data[df_data['eventUri'] != None]
data_event_articles = {event:df_data_events[df_data['eventUri'] == event] for event in all_events}

## VISU
for key in data_event_articles:
    break
    print("For event --> ",key)
    print(data_event_articles[key].shape)
    print("     Exemples de titres..")
    print(data_event_articles[key]['title'])
    print()
    
##

print(data_event_articles["eng-4364137"]['title'][224
])
