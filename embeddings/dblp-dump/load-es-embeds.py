import sys
from elasticsearch import Elasticsearch
from elasticsearch import helpers

doccount = 0
actions = []
es = Elasticsearch("http://ltcpu2:2200")
with open("dblp_biggraph_data_transe/entity_embeddings.tsv") as infile:
    for line in infile:
        items = line.strip().split('\t')
        key = items[0]
        vector = items[1:]
        action = { "_index": "dblpembedstranseindex01", "_source": { "key": key, "embedding": vector } }
        actions.append(action)
        if len(actions) == 100000:
            print("indexing 100k docs ....")
            helpers.bulk(es, actions)
            doccount += 100000
            print("%d done"%(doccount))
            actions = []
helpers.bulk(es, actions)
print("All %d done"%(doccount + len(actions)))
