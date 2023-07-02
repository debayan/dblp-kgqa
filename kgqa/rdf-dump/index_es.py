import json
import gzip
import os
import sys
import glob
import csv
from elasticsearch import Elasticsearch
from elasticsearch import helpers

s = set()
path = sys.argv[1]
index = sys.argv[2]
files = glob.glob(path+'/**/*.gz',recursive=True)
result = []
staging = []
count = 0

doccount = 0
actions = []
es = Elasticsearch('http://localhost:2200')

with open("dblp.labels.nt") as infile:
    for line in infile:
        line = line.strip()
        content = line.split(' ')
        s = content[0]
        p = content[1]
        o = content[2:]
        print(o)


#for file in files:
#    print(file)
#    with gzip.open(file) as f:
#        for line in f:
#            count += 1
#            data = json.loads(line)
#            action = {"_index":index,"_type":"doc","_source":data}
#            actions.append(action)
#            if len(actions) == 10000:
#                print("indexing 10k docs ....")
#                helpers.bulk(es, actions)
#                doccount += 10000
#                print("%d done"%(doccount))
#                actions = []
#helpers.bulk(es, actions)
#print("All %d done"%(doccount + len(actions)))
#
