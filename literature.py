import argparse
import ast
import csv
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple
import gc

import time
from requests.exceptions import ChunkedEncodingError, RequestException

import pandas as pd
import requests
from tqdm import tqdm

def fetch_abstracts(pmids):
    abstracts: List[Tuple[str, str]] = []
    if not pmids:
        return abstracts

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    batch_size = 200
    max_retries = 5
    print(f'{len(pmids)} PMIDs found!\nFetching...')

    for i in tqdm(range(0, len(pmids), batch_size)):
        batch = pmids[i : i + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        }

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(fetch_url, params=params, timeout=20)
                resp.raise_for_status()
                break  # success, exit retry loop
            except (ChunkedEncodingError, RequestException) as e:
                if attempt == max_retries:
                    print(f"[ERROR] Failed after {max_retries} attempts for batch {i}-{i+batch_size}: {e}")
                    continue  # Skip this batch
                wait = 2 ** attempt
                print(f"[WARNING] Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        else:
            continue  # skip to next batch on failure

        try:
            root = ET.fromstring(resp.text)
            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else ""
                abstract = " ".join(
                    t.text or "" for t in article.findall(".//AbstractText")
                )
                if abstract:
                    abstracts.append((pmid, abstract))
        except ET.ParseError as e:
            print(f"[ERROR] XML parse error: {e}")
            continue

    return abstracts


# RAW FETCH
def fetch_pmids_by_string(term: str) -> List[str]:
    """Return a list of PubMed IDs for a given search term."""
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": 1000}
    resp = requests.get(search_url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    pmids = data.get("esearchresult", {}).get("idlist", [])
    return pmids

# NCBI GENE METHOD
def fetch_pmids_by_ncbi_gene_id(term: str) -> str:
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?'
    params = {"db": "gene", "term": f'{term}[PREF] AND Homo sapiens[ORGN]', "usehistory":"y", "retmode": "json"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()

    gene_id = resp.json().get("esearchresult", {}).get("idlist", [])[0]

    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi'
    params = {
        "dbfrom": "gene",
        "db": "pubmed",
        "id": gene_id,
        # "webenv": webenv,
        # "query_key": query_key,
        "retmode": "json"
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    pmids = list(data['PMID'][0:1000]) # TODO: Change the n as needed
    return pmids

# PUBTATOR METHOD (GENE)
def fetch_pmids_by_pubtator3(term: str) -> str:
    # Load Gene Pubtator3 Reference Set
    gene_reference = pd.read_csv('data/pubtator/gene2pubtator3', sep='\t', header=None)
    gene_reference.columns = ['PMID', 'EntityType', 'GeneID', 'MentionText', 'Source']
    print('Gene Pubtator3 set loaded!')

    # Grab PMIDs from Pubtator3 using Reference Set
    gene_hits = gene_reference[gene_reference['MentionText'].str.contains(term, na=False)].reset_index(drop=True)

    pmids = list(gene_hits['PMID'])
    pmids = [str(pmid) for pmid in pmids]

    # Clean up large variables
    del gene_reference
    gc.collect()

    return pmids


# PUBTATOR METHOD (GENE+DRUG)
def fetch_pmids_by_pubtator3drug(gene: str, drugs: List[str]) -> str:
    # Load Gene Pubtator3 Reference Set
    gene_reference = pd.read_csv('data/pubtator/gene2pubtator3', sep='\t', header=None)
    gene_reference.columns = ['PMID', 'EntityType', 'GeneID', 'MentionText', 'Source']
    print('Gene Pubtator3 set loaded!')

    # Load Chemical Pubtator3 Reference Set
    chemical_reference = pd.read_csv('data/pubtator/chemical2pubtator3', sep='\t', header=None)
    chemical_reference.columns = ['PMID', 'EntityType', 'ChemicalID', 'MentionText', 'Source']
    chemical_reference['MentionText'] = chemical_reference['MentionText'].apply(
    lambda x: x.lower() if isinstance(x, str) else ''
)
    print('Drug Pubtator3 set loaded!')

    # Grab PMIDs from Pubtator3 using Reference Set
    gene_hits = gene_reference[gene_reference['MentionText'].str.contains(gene, na=False)].reset_index(drop=True)

    pmid_dict = {}
    for drug in tqdm(drugs):
        drug_hits = chemical_reference[chemical_reference['MentionText'].str.contains(drug.lower(), na=False)].reset_index(drop=True)

        merged_hits = pd.merge(gene_hits, drug_hits, on='PMID', how='inner')
        merged_hits = merged_hits.sort_values(by='PMID', ascending=False)
        merged_hits = merged_hits.drop_duplicates(subset='PMID', keep='first').reset_index(drop=True)

        pmids = list(merged_hits['PMID'])
        pmids = [str(pmid) for pmid in pmids]   

        pmid_dict[drug] = pmids

    # Clean up large variables
    del gene_reference, chemical_reference
    gc.collect() 

    return pmid_dict