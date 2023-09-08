import torch
import os
import re
import json
import requests
import pandas as pd
import numpy as np
import time
import configparser
from transformers import (
        T5ForConditionalGeneration,
        T5Tokenizer,
        BartForConditionalGeneration,
        BartTokenizer
)
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
from fuzzywuzzy import fuzz
# Import utility functions
import sys
sys.path.insert(0, './utils/')
from utils import *

from flask import Flask, request, jsonify

app = Flask(__name__)

config = configparser.ConfigParser()
config.read('config.ini')


#Below is reranker code

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Define your network architecture
        self.embedding = nn.Sequential(
            nn.Linear(969, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward_once(self, x):
        output = self.embedding(x)
        return output

    def forward(self, input_vector):
        output_vector = self.forward_once(input_vector)
        return output_vector

print("Loading reranker models ...")
sentmodel = SentenceTransformer('bert-base-nli-mean-tokens')
reranker_model = {}
es = Elasticsearch("http://banerjee_arefa_dblplink_elasticsearch:9200/")
#es = Elasticsearch("http://localhost:2200/")

reranker_model['transe'] = SiameseNetwork()
reranker_model_transe_path = "models/reranker/model_transe_1/model_epoch_4.pth"
reranker_model['transe'].load_state_dict(torch.load(reranker_model_transe_path))
reranker_model['transe'].eval()

reranker_model['distmult'] = SiameseNetwork()
reranker_model_distmult_path = "models/reranker/model_distmult_1/model_epoch_5.pth"
reranker_model['distmult'].load_state_dict(torch.load(reranker_model_distmult_path))
reranker_model['distmult'].eval()

reranker_model['complex'] = SiameseNetwork()
reranker_model_complex_path = "models/reranker/model_complex_1/model_epoch_6.pth"
reranker_model['complex'].load_state_dict(torch.load(reranker_model_complex_path))
reranker_model['complex'].eval()
print("Loaded reranker models")

def remove_duplicates(array_of_arrays):
    unique_arrays = list(set((inner_array[0],inner_array[1][0],inner_array[1][1]) for inner_array in array_of_arrays))
    unique_arrays_final = [[inner_array[0],[inner_array[1],inner_array[2]]] for inner_array in unique_arrays]
    return unique_arrays_final


def label_search_es(label, enttype):
    try:
        enttypes = ["https://dblp.org/rdf/schema#"+x for x in enttype]
        resp = es.search(index="dblplabelsindex02", query={"bool": {"must": [{"match": {"label": label}}],
                                "filter": {"terms": {"types": enttypes}}}})

        entities = []
        for source in resp['hits']['hits']:
            entities.append([source['_source']['entity'], source['_source']['label'].replace('"','')])
        return entities
    except Exception as err:
        print(err)
        return []

def fetchembedding(entid,embedding):
    try:
        if embedding == 'transe':
            resp = es.search(index="dblpembedstranseindex01", query={"match":{"key":entid}})
        elif embedding == 'distmult':
            resp = es.search(index="dblpembedsdistmultindex01", query={"match":{"key":entid}})
        elif embedding == 'complex':
            resp = es.search(index="dblpembedscomplexindex01", query={"match":{"key":entid}})
        #print(resp)
        embedding = [float(x) for x in resp['hits']['hits'][0]['_source']['embedding']]
        return embedding
    except Exception as err:
        print(err)
        return []

def link(question, label, entity_type, modeltype='transe'):
    candidate_entities_labels = label_search_es(label, entity_type)
    candidate_embeddings = [fetchembedding(x[0], modeltype) for x in candidate_entities_labels]
    question_encoding = list(sentmodel.encode([question])[0])+ 201*[0.0]
    question_embedding = reranker_model[modeltype](torch.tensor(question_encoding))

    candidate_encodings = [list(sentmodel.encode([x[1]])[0])+candidate_embeddings[idx]+[fuzz.token_set_ratio(x[1],question)/100.0]  for idx,x in enumerate(candidate_entities_labels)]
    candidate_embeddings = [reranker_model[modeltype](torch.tensor(x)) for x in candidate_encodings]
    arr = []
    for idx,candidate_embedding in enumerate(candidate_embeddings):
        distance = torch.norm(question_embedding - candidate_embedding, p=2)
        arr.append([distance.item(),candidate_entities_labels[idx]])
    arr = remove_duplicates(arr)
    sorted_entities =  sorted(arr, key=lambda d: d[0])
    print(sorted_entities)
    return sorted_entities

#Above this is reranker code
#Below is label span model and code


def setup_models(model_names):
    models = {}
    tokenizers = {}
    for model_name in model_names:
        model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict = True)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        models[model_name] = model
        tokenizers[model_name] = tokenizer
        output_dir = './models/labelspan/'
        load_model(output_dir, models[model_name], 'model_{}.pth'.format(model_name))
    return models, tokenizers

def infer(question, _model, _tokenizer):
    # Tokenize
    input_question = _tokenizer.encode_plus(question, padding=True, truncation=True, return_tensors='pt')
    # Generate answer containing label and type
    output = _model.generate(input_ids=input_question['input_ids'], attention_mask=input_question['attention_mask'], max_length=512)
    predicted = [_tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in output]
    return predicted

def separate_ents(predicted):
    ents = predicted.split("[SEP]")
    return ents[:-1]

def get_labels_types(ents):
    labels = []
    types =[]
    for ent in ents:
        all_types = []
        if ":" in ent:
            i = ent.rindex(":")
            labels.append(ent[:i].strip())
            temp = ent[i+1:]
            all_types = temp.split(',')
            all_types = [item.strip() for item in all_types[:-1]]
            types.append(all_types)
    print("label and types: ")
    print(labels)
    print(types)
    return labels, types


models_names = ["t5-small", "t5-base"]
print("Loading labelspan models ...")
models, tokenizers = setup_models(models_names)
print("Models loaded.")

@app.route('/api/entitylinker/<modelname>/<embedding>', methods=['POST'])
def receive_json(modelname, embedding):
    try:
        # Get the JSON data from the request
        data = request.get_json()
        # Check if the request contains JSON data
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        question = data["question"]
        predicted = infer(question, models[modelname],tokenizers[modelname])
        ents = separate_ents(predicted[0])
        allresults = []
        labels, types = get_labels_types(ents)
        for label,typ in zip(labels,types):
            results = link(question, label, typ, embedding)
            allresults.append({'result':results, 'label': label, 'type':typ})
        # Process the JSON data as needed
        # For demonstration purposes, let's just return the received JSON
        print(jsonify({'question':question,'predictedlabelspans':predicted, 'entitylinkingresults':allresults}))
        return jsonify({'question':question,'predictedlabelspans':predicted, 'entitylinkingresults':allresults}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)

