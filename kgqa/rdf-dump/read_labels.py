import sys,os,json

d = {}

imprels = ['<https://dblp.org/rdf/schema#primaryFullCreatorName>','<https://dblp.org/rdf/schema#title>','<http://www.w3.org/2000/01/rdf-schema#label>']

f = open(sys.argv[1],'w')

with open("dblp.nt") as infile:
    for line in infile:
        line = line.strip()
        if not line:
            continue
        for rel in imprels:
            if rel in line:
                f.write(line+'\n')
#        content = line.split(' ')
#        p = content[1]
#        if p in imprels:
#            s = content[0]
#            if s not in d:
#                d[s] = {'label': content[2], 'relation':p}
f.close()
