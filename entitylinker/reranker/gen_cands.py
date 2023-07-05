import sys,os,json
from elasticsearch import Elasticsearch
import random


es = Elasticsearch("http://ltcpu2:2200")

def label_search_es(label):
    try:
        resp = es.search(index="dblplabelsindex01", query={"match":{"label":{"query":label}}})
        #print(resp)
        entities = []
        for source in resp['hits']['hits']:
            entities.append(source['_source']['entity'])
        return entities
    except Exception as err:
        print(err)
        return []

def fetchembedding(entid):
    try:
        resp = es.search(index="dblpembedsindex01", query={"match":{"key":entid}})
        #print(resp)
        embedding = resp['hits']['hits'][0]['_source']['embedding']
        return embedding
    except Exception as err:
        print(err)
        return []

def getlabels(entid):
    try:
        resp = es.search(index="dblplabelsindex01", query={"match":{"entity":entid}})
        #print(resp)
        labels = []
        for source in resp['hits']['hits']:
            labels.append(source['_source']['label'])
        return labels
    except Exception as err:
        print(err)
        return []

def remove_all_occurrences(lst, item):
    while item in lst:
        lst.remove(item)

d = json.loads(open(sys.argv[1]).read())

f = open(sys.argv[2],'w')
for item in d['questions']:
    try:
        entities = item['entities']
        print(item['id'])
        print(item['question']['string'])
        print(entities)
        newents = []
        for ent in entities:
            res = getlabels(ent)
            entlabel = eval(res[0])
            print("res:",entlabel)
            cands = label_search_es(entlabel)
            remove_all_occurrences(cands,ent)
            negative_sample = random.choice(cands)
            print("gold ent:",ent)
            print("neg  ent:",negative_sample)
            goldemb = fetchembedding(ent)
            negemb = fetchembedding(negative_sample)
            neglabel  = eval(getlabels(negative_sample)[0])
            newents.append({'goldent':ent, 'goldemb':goldemb, 'goldlabel':entlabel, 'negent':negative_sample, 'negemb':negemb , 'neglabel':neglabel})
        newitem = {'id':item['id'], 'question':item['question'], 'entity_samples': newents}
        f.write(json.dumps(newitem)+'\n')
    except Exception as err:
        print(err)
f.close()
