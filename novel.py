# Imports & NLP Code
import pandas as pd
from transformers import pipeline
import pandas as pd
import requests
from tqdm import tqdm
import inflect  

PIPE_GENE = pipeline("token-classification", model="alvaroalon2/biobert_genetic_ner", aggregation_strategy="first")

PIPE_CHEMICAL = pipeline("token-classification", model="alvaroalon2/biobert_chemical_ner", aggregation_strategy="first")

PIPE_DISEASE = pipeline("token-classification", model="alvaroalon2/biobert_diseases_ner", aggregation_strategy="first")


def process_text(text):    
    tdf1 = _tag_genes(text)
    tdf2 = _tag_chemicals(text)
    tdf3 = _tag_diseases(text)
    df = pd.concat([tdf1,tdf2,tdf3])
    df = _drop_unknowns(df)
    df = normalize(df)
    return df

def batch(corpus):
    df = pd.DataFrame()
    for entry in tqdm(corpus):
        tdf1 = _tag_genes(entry)
        tdf2 = _tag_chemicals(entry)
        tdf3 = _tag_diseases(entry)
        df = pd.concat([df,tdf1,tdf2,tdf3])
    df = _drop_unknowns(df)
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    # df = normalize(df)
    return df

def normalize(result):
    result['concept_match_type'] = None
    result['concept_id'] = None
    result['concept_label'] = None
    for index, row in tqdm(result.iterrows()):
        word = _singularize(row['word'])
        if row['entity_group'] == 'GENETIC':
            norm_result = _normalize_gene(word)
        if row['entity_group'] == 'CHEMICAL':
            norm_result = _normalize_therapy(word)
        if row['entity_group'] == 'DISEASE':
            norm_result = _normalize_disease(word)
        result.at[index, 'concept_match_type'] = norm_result[0]
        result.at[index, 'concept_id'] = norm_result[1]
        result.at[index, 'concept_label'] = norm_result[2]
    return result


def _singularize(word):
    inflector = inflect.engine()
    return inflector.singular_noun(word) or word

# TODO: Implement cached queries to improve normalization time -- Brian Walsh/ Kori / James
def _normalize_gene(word):
    r = requests.get(f'https://normalize.cancervariants.org/gene/normalize?q={word}')
    response = r.json()
    try:
        if response['match_type'] != 0:
            match_type = response['match_type']
            concept_id = response['gene']['id']
            label = response['gene']['name']
        else:
            match_type = response['match_type']
            concept_id = None
            label = None
    except:
        match_type = 'Failure to Normalize'
        concept_id = 'Failure to Normalize'
        label = 'Failure to Normalize'

    return [match_type, concept_id, label]

def _normalize_disease(word):
    r = requests.get(f'https://normalize.cancervariants.org/disease/normalize?q={word}')
    response = r.json()
    try:
        if response['match_type'] != 0:
            match_type = response['match_type']
            concept_id = response['disease']['id']
            label = response['disease']['name']
        else:
            match_type = response['match_type']
            concept_id = None
            label = None
    except:
        match_type = 'Failure to Normalize'
        concept_id = 'Failure to Normalize'
        label = 'Failure to Normalize'

    return [match_type, concept_id, label]

def _normalize_therapy(word):
    r = requests.get(f'https://normalize.cancervariants.org/therapy/normalize?q={word}&infer_namespace=true')
    response = r.json()
    try:
        if response['match_type'] != 0:
            match_type = response['match_type']
            concept_id = response['therapy']['id']
            label = response['therapy']['name']
        else:
            match_type = response['match_type']
            concept_id = None
            label = None
    except:
        match_type = 'Failure to Normalize'
        concept_id = 'Failure to Normalize'
        label = 'Failure to Normalize'

    return [match_type, concept_id, label]


def _drop_unknowns(result):
    try:
        dropped = result[result['entity_group']!='0'].reset_index(drop=True)
    except:
        dropped = result
    return dropped

def _tag_genes(text):
    gene_results = PIPE_GENE(text)
    df = _drop_unknowns(pd.DataFrame(gene_results))
    df['original_text'] = text
    return df

def _tag_chemicals(text):
    chem_results = PIPE_CHEMICAL(text)
    df = _drop_unknowns(pd.DataFrame(chem_results))
    df['original_text'] = text
    return df

def _tag_diseases(text):
    disease_results = PIPE_DISEASE(text)
    df = _drop_unknowns(pd.DataFrame(disease_results))
    df['original_text'] = text
    return df