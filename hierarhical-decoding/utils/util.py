import random
import numpy as np
import torch
import torch.nn as nn
import re


def initialize_weights(m):
	'''
		Initialize the weights of a model
		m: model
	'''
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def set_seed(seed):
	'''
		Setting a seed to make our experiments reproducible
		seed: seed value
	'''
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True

def count_parameters(model):
	'''
		Count the number of parameters that requires grad
		model: model
	'''

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def split_triples(text):
    triples, triple = [], []
    for w in text:
        if w not in ['<TRIPLE>', '</TRIPLE>']:
            triple.append(w)
        elif w == '</TRIPLE>':
            triples.append(triple)
            triple = []
    return triples

def join_triples(triples):
    result = []
    for triple in triples:
        result.append('<TRIPLE>')
        result.extend(triple)
        result.append('</TRIPLE>')
    return result

def delexicalize(triples):
    entities = {}
    entity_pos = 1
    for triple in triples:
        agent = triple[0]
        if agent not in entities:
            entities[agent] = 'ENTITY-' + str(entity_pos)
            entity_pos += 1
        triple[0] = entities[agent]

        patient = triple[-1]
        if patient not in entities:
            entities[patient] = 'ENTITY-' + str(entity_pos)
            entity_pos += 1
        triple[-1] = entities[patient]

    return triples

def entity_mapping(triples):
    entitytag = {}
    entities = {}
    entity_pos = 1
    for triple in triples:
        agent = triple[0]
        if agent not in entitytag:
            entitytag[agent] = 'ENTITY-' + str(entity_pos)
            entities['ENTITY-' + str(entity_pos)] = agent
            entity_pos += 1

        patient = triple[-1]
        if patient not in entitytag:
            entitytag[patient] = 'ENTITY-' + str(entity_pos)
            entities['ENTITY-' + str(entity_pos)] = patient
            entity_pos += 1

    return entities

def split_struct(text):
    sentences, triples, triple = [], [], []
    for w in text:
        if w not in ['<SNT>', '</SNT>', '<TRIPLE>', '</TRIPLE>']:
            triple.append(w)
        elif w == '</TRIPLE>':
            triples.append(triple)
            triple = []
        elif w == '</SNT>':
            sentences.append(triples)
            triples = []
    return sentences

def join_struct(sentences):
    result = []
    for sentence in sentences:
        result.append('<SNT>')
        for triple in sentence:
            result.append('<TRIPLE>')
            result.extend(triple)
            result.append('</TRIPLE>')
        result.append('</SNT>')
    return result

def delexicalize_struct(struct):
    entities, entity_pos = {}, 1
    for sentence in struct:
        for triple in sentence:
            agent = triple[0]
            if agent not in entities:
                entities[agent] = 'ENTITY-' + str(entity_pos)
                entity_pos += 1
            triple[0] = entities[agent]

            patient = triple[-1]
            if patient not in entities:
                entities[patient] = 'ENTITY-' + str(entity_pos)
                entity_pos += 1
            triple[-1] = entities[patient]

    return struct

def delexicalize_verb(template):
    regex = r'(tense=|person=)(.*?),'
    template = re.sub(regex, r'\1null,', template)

    regex = r'(number=)(.*?)]'
    return re.sub(regex, r'\1null]', template)

