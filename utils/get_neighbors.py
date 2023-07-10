# install sparql wrapper for working with Python
!pip install -q sparqlwrapper

from SPARQLWrapper import SPARQLWrapper, JSON, XML, N3, RDF

# determining sparql endpoint
sparql = SPARQLWrapper("https://dblp-kg.ltdemos.informatik.uni-hamburg.de/sparql")

def get_neighbors(entity):
    query_prefix = """
        PREFIX dbo: <http://dbpedia.org/ontology/> 
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        PREFIX : <http://dbpedia.org/resource/>
        PREFIX dbpedia2: <http://dbpedia.org/property/>
        PREFIX dbpedia: <http://dbpedia.org/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    """
    # set return type to JSON
    sparql.setReturnFormat(JSON)
    
    # 1. for entity => node relations
    sparql.setQuery(
        query_prefix
        +
        """
        SELECT distinct ?e ?p ?node WHERE{
            ?e rdfs:label "%s" .
            ?e ?p ?node .
        }
        LIMIT 50
        """%(given_entity)
    )
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        print(result["e"]["value"] + " : " + result["p"]["value"] + " = " + result["node"]["value"])
    print('\n\n')

    # 2. for node => entity relations
    sparql.setQuery(
        query_prefix
        +
        """
        SELECT distinct ?node ?p ?e WHERE{
            ?e rdfs:label "%s" .
            ?node ?p ?e .
        }
        LIMIT 50
        """%(given_entity)
    )
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        print(result["node"]["value"] + " : " + result["p"]["value"] + " = " + result["e"]["value"])

# take entity label as input
given_entity = input('Enter entity label:\n')
print(given_entity)

get_neighbors(given_entity)
