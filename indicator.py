import re
from typing import Tuple, List
import ast
from datetime import datetime
import csv
import shutil
from tqdm import tqdm
import os

def parse_drug_terms(drug_entry: str) -> List[str]:
    """Return all names from a ``(brand, generic)`` tuple string."""
    try:
        value = ast.literal_eval(drug_entry)
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value if isinstance(v, str)]
    except Exception:
        pass
    return [drug_entry]


# Words Sets & Intended Weights
CLINICAL_STUDY = {'randomized', 'randomised', 'placebo', 'double-blind', 'single-blind', 'controlled', 'trial', 'pragmatic trial', 'phase i', 'phase ii', 'phase iii', 'phase iv', 'clinical study', 'clinical trial', 'multicenter', 'interventional', 'crossover', 'parallel group', 'allocation', 'intention-to-treat', 'efficacy', 'endpoint', 'primary outcome', 'progression free survival', 'overall survival'} # Weight = 1

CASE_REPORT = {'case report', 'case reports', 'case series', 'case study', 'case studies', 'single patient', 'n-of-1'} # Weight = 1

ANIMAL_EVIDENCE = {'in vivo', 'xenograft', 'patient-derived xenograft', 'PDX', 'orthotopic', 'murine', 'mouse', 'mice', 'rat', 'rats', 'zebrafish', 'Drosophila', 'C. elegans', 'animal model', 'live animal'} # Weight = 0.5

CELL_LINES = {'in vitro', 'cell line', 'cultured cells', 'cell culture', 'monolayer', '3D culture', 'organoid', 'primary cells', 'immortalized cell line', 'transfected cell line'} # Weight = 0.5

IMAGING_EVIDENCE = {'MRI', 'magnetic resonance imaging', 'CT', 'computed tomography', 'PET', 'positron emission tomography', 'ultrasound', 'radiography', 'fluorescence microscopy', 'confocal microscopy', 'histopathology', 'histological imaging', 'digital image analysis', 'quantitative imaging', 'imaging biomarkers'} # Weight = 0.5

RETROSPECTIVE_STUDY = {'retrospective study', 'retrospective analysis', 'retrospective review', 'chart review', 'registry data', 'historical cohort', 'medical record review', 'retrospective cohort'} # Weight = 0.75

CHEMOTHERAPY_AGENTS = {
    # Alkylating agents
    "Cyclophosphamide", "Carmustine", "Cisplatin", "Carboplatin", "Oxaliplatin",
 
    # Anthracyclines / related
    "Doxorubicin", "Daunorubicin", "Daunomycin", "Idarubicin",
 
    # Antimetabolites (nucleoside analogues)
    "Fluorouracil", "Methotrexate", "Azacitidine",
 
    # Taxanes & microtubule disruptors
    "Paclitaxel", "Docetaxel", "Vincristine", "Vinblastine", "Vinorelbine",
 
    # Topoisomerase inhibitors
    "Etoposide", "Topotecan", "Irinotecan",
} # Weight = 0, Should be hidable 

def analyze_relation(drug: str, gene: str, abstract: str) -> Tuple[str, float]:
    """
    """

    text = abstract.lower()
    drug_present = re.search(re.escape(drug.lower()), text) is not None
    gene_present = re.search(re.escape(gene.lower()), text) is not None

    if not (drug_present and gene_present):
        return ('not_evaluated', 0.0)

    indicators = {}
    
    indicators['clinical_study'] = sum(word in text for word in CLINICAL_STUDY)
    indicators['case_report'] = sum(word in text for word in CASE_REPORT)
    indicators['animal_evidence'] = sum(word in text for word in ANIMAL_EVIDENCE)
    indicators['cell_line'] = sum(word in text for word in CELL_LINES)
    indicators['imaging_evidence'] = sum(word in text for word in IMAGING_EVIDENCE)
    indicators['retrospective'] = sum(word in text for word in RETROSPECTIVE_STUDY)
    indicators['unweighted_total'] = sum(indicators.values())

    if indicators['unweighted_total'] > 0:
        label = 'indicator_evidence'
    else:
        label = 'no_indicator_evidence'

    return label, indicators

def generate_indicators(abstracts, reference_df, start=0, stop=-1):
    gene = reference_df['Gene'][0]

    timestamp = datetime.now().strftime("%Y-%m-%d")
    timestamp_folder = f'{timestamp}_{gene}'

    for idx, row in tqdm(reference_df.iterrows()):
        gene = str(row.get('Gene'))
        drug = str(row.get('Drug'))
        drug = parse_drug_terms(drug)

        results = []
        for pmid, abstract in tqdm(abstracts[start:stop], desc="Abstracts", leave=False):
            label, scores = analyze_relation(drug[0], gene, abstract)
            results.append({"pmid": pmid, "label": label, "scores": scores })

            if results: 
                os.makedirs(timestamp_folder, exist_ok=True)
                out_filename = f'{gene}_{drug[0]}.csv'.replace("/", "-")
                out_path = os.path.join(timestamp_folder, out_filename)
                with open(out_path, "w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=["pmid", "label", "scores"])
                    writer.writeheader()
                    writer.writerows(results)
                    
    archive_path = shutil.make_archive(base_name=timestamp_folder, format="zip", base_dir=f'{timestamp}_{gene}')
    shutil.rmtree(timestamp_folder)
    print(f'Results saved to {timestamp_folder}.zip!')

