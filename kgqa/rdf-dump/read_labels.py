import sys,os,json

d = {}

imprels = ['<https://dblp.org/rdf/schema#primaryFullCreatorName>','<https://dblp.org/rdf/schema#title>','<http://www.w3.org/2000/01/rdf-schema#label>']
types = ['<http://purl.org/spar/datacite/ResourceIdentifier>', '<https://dblp.org/rdf/schema#Data>', '<http://www.w3.org/2002/07/owl#SymmetricProperty>', '<http://www.w3.org/1999/02/22-rdf-syntax-ns#Property>', '<https://dblp.org/rdf/schema#Reference>', '<https://dblp.org/rdf/schema#Book>', '<https://dblp.org/rdf/schema#Publication>', '<http://purl.org/spar/datacite/PersonalIdentifier>', '<http://www.w3.org/1999/02/22-rdf-syntax-ns#List>', '<https://dblp.org/rdf/schema#Group>', '<https://dblp.org/rdf/schema#Inproceedings>', '<http://purl.org/spar/datacite/Identifier>', '<https://dblp.org/rdf/schema#Creator>', '<https://dblp.org/rdf/schema#Informal>', '<https://dblp.org/rdf/schema#Withdrawn>', '<https://dblp.org/rdf/schema#Editorship>', '<http://www.w3.org/2002/07/owl#TransitiveProperty>', '<http://www.w3.org/2000/01/rdf-schema#Class>', '<https://dblp.org/rdf/schema#Person>', '<https://dblp.org/rdf/schema#Incollection>', '<https://dblp.org/rdf/schema#Article>', '<https://dblp.org/rdf/schema#AmbiguousCreator>']

with open("dblp.nt") as infile:
    for line in infile:
        line = line.strip()
        if not line:
            continue
        #print("line:",line)
        content = line.split(' ')
        p = content[1]
        if p in imprels:
            s = content[0]
            if s not in d:
                d[s] = {'label': content[2], 'relation':p}

