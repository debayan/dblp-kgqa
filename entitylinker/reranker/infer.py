import torch
import sys
import os
import json
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
from fuzzywuzzy import fuzz

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

es = Elasticsearch("http://ltcpu2:2200/")
model = SiameseNetwork()
model_path = sys.argv[1]
model.load_state_dict(torch.load(model_path))
model.eval()
sentmodel = SentenceTransformer('bert-base-nli-mean-tokens')


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

def fetchembedding(entid):
    try:
        resp = es.search(index="dblpembedsindex01", query={"match":{"key":entid}})
        #print(resp)
        embedding = [float(x) for x in resp['hits']['hits'][0]['_source']['embedding']]
        return embedding
    except Exception as err:
        print(err)
        return []


def link(question, label, entity_type):
    candidate_entities_labels = label_search_es(label, entity_type)
    candidate_embeddings = [fetchembedding(x[0]) for x in candidate_entities_labels]
    question_encoding = list(sentmodel.encode([question])[0])+ 201*[0.0]
    question_embedding = model(torch.tensor(question_encoding))

    candidate_encodings = [list(sentmodel.encode([x[1]])[0])+candidate_embeddings[idx]+[fuzz.token_set_ratio(x[1],question)/100.0]  for idx,x in enumerate(candidate_entities_labels)]
    candidate_embeddings = [model(torch.tensor(x)) for x in candidate_encodings]
    arr = []
    for idx,candidate_embedding in enumerate(candidate_embeddings):
        distance = torch.norm(question_embedding - candidate_embedding, p=2)
        arr.append([distance.item(),candidate_entities_labels[idx]])
    sorted_entities =  sorted(arr, key=lambda d: d[0])
    print(sorted_entities)
    return sorted_entities
      
    
        

app = Flask(__name__)

@app.route('/entitylinker', methods=['POST'])
def process():
    data = request.get_json()  # Get the JSON data from the request
    question = data['question']
    label = data['label']
    entity_type = data['type']
    results = link(question, label, entity_type)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0',port=5001)

