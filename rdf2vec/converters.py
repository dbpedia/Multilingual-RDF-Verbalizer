from graph import KnowledgeGraph, Vertex
from tqdm import tqdm
from os import listdir
import rdflib

noisy_urls = ["http://dbpedia.org/resource/", "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "http://www.w3.org/2002/07/owl#", "http://dbpedia.org/ontology/", "http://xmlns.com/foaf/0.1/", "http://www.w3.org/2000/01/rdf-schema#", "http://purl.org/dc/terms/", "http://dbpedia.org/datatype/"]


def clean_url(url):
    """
        Removes "<", ">" and urls to obtain only the node name
    """
    for noisy_url in noisy_urls:
        url = str(url).replace(noisy_url,"").replace("<", "").replace(">", "").lower()
    return url

def create_kg(kg, triples):
    """
        Builds the Knowledge graph from the triples of a dbpedia file. 
    """
    for (s, p, o) in tqdm(triples):

        s = clean_url(s)
        o = clean_url(o)
        p = clean_url(p)
        
        s_v = Vertex(str(s))
        o_v = Vertex(str(o))
        p_v = Vertex(str(p), predicate=True, _from=s_v, _to=o_v)
        kg.add_vertex(s_v)
        kg.add_vertex(p_v)
        kg.add_vertex(o_v)
        kg.add_edge(s_v, p_v)
        kg.add_edge(p_v, o_v)

def load_kg(files, filetype=None, strategy="rnd"):
    """Convert a rdflib.Graph (located at file) to our KnowledgeGraph."""
    kg = KnowledgeGraph(strategy=strategy)

    for f in files:
        g = rdflib.Graph()
        if filetype is not None:
            g.parse(f, format=filetype)
        else:
            g.parse(f)

        create_kg(kg, g)

    return kg

