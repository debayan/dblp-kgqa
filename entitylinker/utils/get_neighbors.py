# install sparql wrapper for working with Python
!pip install -q sparqlwrapper

from SPARQLWrapper import SPARQLWrapper, JSON, XML, N3, RDF

# determining sparql endpoint
sparql = SPARQLWrapper("https://dbpedia.org/sparql")

def get_neighbors(entity):
    # 1. for 1:N relations
    sparql.setQuery(
        """
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

        SELECT distinct ?p ?node WHERE{
            :""" + given_entity + """ ?p ?node .
            ?node ?p :""" + given_entity + """ .
        }
        LIMIT 50
        """
    )

    # set return type to JSON
    sparql.setReturnFormat(JSON)

    # execcute sparql query and write result to results
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        print(result["p"]["value"] + " = " + result["node"]["value"])

    print('\n\n')

    # 2. for N:1 relations
    sparql.setQuery(
        """
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

        SELECT distinct ?node ?p WHERE{
            ?node ?p :""" + given_entity + """ .
        }
        LIMIT 50
        """
    )

    # set return type to JSON
    sparql.setReturnFormat(JSON)

    # execcute sparql query and write result to results
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        print(result["node"]["value"] + " = " + result["p"]["value"])

# take entity label as input
given_entity = input('Enter entity label:\n')
print(given_entity)

get_neighbors(given_entity)
