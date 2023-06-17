import sys,os,json,requests

alltypes = ['<http://purl.org/spar/datacite/ResourceIdentifier>', '<https://dblp.org/rdf/schema#Data>', '<http://www.w3.org/2002/07/owl#SymmetricProperty>', '<http://www.w3.org/1999/02/22-rdf-syntax-ns#Property>', '<https://dblp.org/rdf/schema#Reference>', '<https://dblp.org/rdf/schema#Book>', '<https://dblp.org/rdf/schema#Publication>', '<http://purl.org/spar/datacite/PersonalIdentifier>', '<http://www.w3.org/1999/02/22-rdf-syntax-ns#List>', '<https://dblp.org/rdf/schema#Group>', '<https://dblp.org/rdf/schema#Inproceedings>', '<http://purl.org/spar/datacite/Identifier>', '<https://dblp.org/rdf/schema#Creator>', '<https://dblp.org/rdf/schema#Informal>', '<https://dblp.org/rdf/schema#Withdrawn>', '<https://dblp.org/rdf/schema#Editorship>', '<http://www.w3.org/2002/07/owl#TransitiveProperty>', '<http://www.w3.org/2000/01/rdf-schema#Class>', '<https://dblp.org/rdf/schema#Person>', '<https://dblp.org/rdf/schema#Incollection>', '<https://dblp.org/rdf/schema#Article>', '<https://dblp.org/rdf/schema#AmbiguousCreator>']
allrelations = ["<https://dblp.org/rdf/schema#primaryFullCreatorName>", "<https://dblp.org/rdf/schema#editedBy>", "<https://dblp.org/rdf/schema#publishedInJournalVolumeIssue>", "<http://www.w3.org/2000/01/rdf-schema#range>", "<https://dblp.org/rdf/schema#authoredBy>", "<https://dblp.org/rdf/schema#publishedIn>", "<https://dblp.org/rdf/schema#otherFullCreatorName>", "<https://dblp.org/rdf/schema#title>", "<https://dblp.org/rdf/schema#creatorNote>", "<http://purl.org/dc/terms/license>", "<http://purl.org/spar/datacite/usesIdentifierScheme>", "<http://www.w3.org/2002/07/owl#equivalentClass>", "<http://www.w3.org/2000/01/rdf-schema#subPropertyOf>", "<https://dblp.org/rdf/schema#publishedBy>", "<https://dblp.org/rdf/schema#monthOfPublication>", "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>", "<https://dblp.org/rdf/schema#publishedInSeriesVolume>", "<https://dblp.org/rdf/schema#otherHomepage>", "<http://www.w3.org/2000/01/rdf-schema#subClassOf>", "<http://www.w3.org/2000/01/rdf-schema#domain>", "<https://dblp.org/rdf/schema#orcid>", "<http://www.w3.org/2002/07/owl#equivalentProperty>", "<https://dblp.org/rdf/schema#yearOfEvent>", "<https://dblp.org/rdf/schema#webpage>", "<https://dblp.org/rdf/schema#listedOnTocPage>", "<http://purl.org/spar/datacite/hasIdentifier>", "<http://purl.org/spar/literal/hasLiteralValue>", "<https://dblp.org/rdf/schema#publishedInJournal>", "<https://dblp.org/rdf/schema#otherElectronicEdition>", "<https://dblp.org/rdf/schema#awardWebpage>", "<https://dblp.org/rdf/schema#publishedInSeries>", "<https://dblp.org/rdf/schema#archivedWebpage>", "<https://dblp.org/rdf/schema#publicationNote>", "<https://dblp.org/rdf/schema#thesisAcceptedBySchool>", "<https://dblp.org/rdf/schema#primaryElectronicEdition>", "<http://purl.org/dc/terms/modified>", "<https://dblp.org/rdf/schema#doi>", "<https://dblp.org/rdf/schema#wikidata>", "<https://dblp.org/rdf/schema#otherAffiliation>", "<https://dblp.org/rdf/schema#primaryHomepage>", "<https://dblp.org/rdf/schema#bibtexType>", "<https://dblp.org/rdf/schema#archivedElectronicEdition>", "<https://dblp.org/rdf/schema#pagination>", "<https://dblp.org/rdf/schema#publishedInJournalVolume>", "<https://dblp.org/rdf/schema#yearOfPublication>", "<https://dblp.org/rdf/schema#publishedInBookChapter>", "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>", "<http://www.w3.org/2000/01/rdf-schema#comment>", "<http://www.w3.org/2002/07/owl#inverseOf>", "<http://www.w3.org/2000/01/rdf-schema#label>", "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>", "<http://purl.org/dc/terms/creator>", "<https://dblp.org/rdf/schema#isbn>", "<https://dblp.org/rdf/schema#publishersAddress>", "<https://dblp.org/rdf/schema#publishedInBook>", "<https://dblp.org/rdf/schema#orderedCreators>", "<http://www.w3.org/2002/07/owl#differentFrom>", "<https://dblp.org/rdf/schema#publishedAsPartOf>", "<https://dblp.org/rdf/schema#wikipedia>", "<https://dblp.org/rdf/schema#numberOfCreators>", "<http://www.w3.org/2002/07/owl#sameAs>", "<https://dblp.org/rdf/schema#primaryAffiliation>" ]

labelrels = ['<https://dblp.org/rdf/schema#primaryFullCreatorName>','<https://dblp.org/rdf/schema#title>','<http://www.w3.org/2000/01/rdf-schema#label>']

types_relations = list(set(alltypes+allrelations))

masterdict = {}

for id,rel in enumerate(types_relations):
    masterdict[rel] = '<extra_id_'+str(id)+'>'

inv_masterdict = {v: k for k, v in masterdict.items()}


def sparqlendpoint(query):
    try:
        url = 'http://ltdocker:8897/sparql'
        query =  query
        #print(query)
        headers = {'Accept':'application/sparql-results+json'}
        r = requests.get(url, headers=headers, params={'format': 'json', 'query': query})
        json_format = r.json()
        #print(json_format)
        results = json_format
        return results
    except Exception as err:
        print(err)
        return ''

def fetchtype(ent):
    try:
        url = 'http://ltdocker:8897/sparql'
        query =  '''select ?type where { %s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?type}'''%(ent)
        headers = {'Accept':'application/sparql-results+json'}
        r = requests.get(url, headers=headers, params={'format': 'json', 'query': query})
        json_format = r.json()
        #print(json_format)
        results = json_format
        return results
    except Exception as err:
        print(err)
        return ''

def fetchlabel(ent):
    labels = []
    url = 'http://ltdocker:8897/sparql'
    for labelrel in labelrels:
        query =  '''select ?type where { %s %s ?type}'''%(ent,labelrel)
        headers = {'Accept':'application/sparql-results+json'}
        r = requests.get(url, headers=headers, params={'format': 'json', 'query': query})
        json_format = r.json()
        try:
            labels.append(json_format['results']['bindings'][0]['type']['value'])
        except Exception as err:
            print(err)
    return labels

d = json.loads(open(sys.argv[1]).read())

outarr = []
f = open(sys.argv[2],'w')
for item in d['questions']:
    q = item['query']['sparql']
    print(item['id'])
    print(item['question']['string'])
    print(item['paraphrased_question']['string'])
    #print(sparqlendpoint(q))
    for ent in item['entities']:
        types = fetchtype(ent)
        typarr = []
        try:
            for typ in types['results']['bindings']:
                typarr.append(typ['type']['value'])
        except Exception as err:
            print(err)
            break
        print(typarr)
        labels = fetchlabel(ent)
        labels = list(set(labels))
        print(labels)
        try:
            q = q.replace(ent,'<ent> '+labels[0]+' : <'+typarr[0]+'> </ent>')
        except Exception as err:
            print(err)
            break
        print(item['query']['sparql'])
        print(q)
        for k,v in masterdict.items():
            k1 = k.replace('https://','prefix@@')
            q = q.replace(k,k1)
        print(q)
        print("------------------------------------------------")
    q = q.replace('(',' ( ').replace(')',' ) ')
    f.write(json.dumps({'question':item['question']['string'], 'query':q})+'\n')
    f.write(json.dumps({'question':item['paraphrased_question']['string'], 'query':q})+'\n')
#    sys.exit(1)

f.close()
