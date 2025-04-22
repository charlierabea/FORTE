import pandas as pd
import json

def extract_keywords_from_text(text, json_data):
    text = f" {text.lower()} "
    keywords = set()
    for key in json_data:
        if key in text:
            keywords.add(key.strip())    
    return list(keywords)

def map_to_synonyms(key, json_data):
    for main_key, synonyms in json_data.items():
        # print(synonyms)
        if key in synonyms:
            # print(key)
            return set(synonyms)
    return {key}  # Return a set with the key itself

def map_to_representative_synonym(key, json_data):
    for main_key, synonyms in json_data.items():
        if key in synonyms:
            return main_key  # Return the primary keyword
    return key

def calculate_precision_recall(gt_set, out_set, json_data):
    if not gt_set and not out_set:
        return None, None
    if not gt_set:
        return 0, None
    if not out_set:
        return None, 0
    # Pre-process keywords to their representative synonyms
    gt_representative = {map_to_representative_synonym((' ' + keyword), json_data) for keyword in gt_set}
    out_representative = {map_to_representative_synonym((' ' + keyword), json_data) for keyword in out_set}

    # True positive calculation considering synonyms
    tp = sum(1 for gt_key in gt_representative if any(gt_key in out_synonyms for out_synonyms in map(lambda x: map_to_synonyms((x), json_data), out_representative)))
    
    # print(tp)

    fp = len(out_representative) - tp
    fn = len(gt_representative) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / len(gt_representative) if gt_representative else 0

    return precision, recall


def process_excel_sheet(sheet_df, json_data):
    for col_prefix in ['degree', 'landmark', 'feature', 'impression']:
        sheet_df[f'gt_{col_prefix}'] = sheet_df['gt'].apply(lambda x: list(set(extract_keywords_from_text(x, json_data[col_prefix]))))
        sheet_df[f'out_{col_prefix}'] = sheet_df['parsed_output'].apply(lambda x: list(set(extract_keywords_from_text(x, json_data[col_prefix]))))
        
        sheet_df[f'prec_{col_prefix}'], sheet_df[f'recall_{col_prefix}'] = zip(*sheet_df.apply(lambda row: calculate_precision_recall(set(row[f'gt_{col_prefix}']), set(row[f'out_{col_prefix}']), json_data[col_prefix]), axis=1))


def process_excel(filename, json_data):
    xls = pd.read_excel(filename, sheet_name='ABC', engine='openpyxl')
    process_excel_sheet(xls, json_data)
    xls.to_excel("./excel_files/FORTE_evaluated.xlsx", index=False)
    

# Load JSON data
with open('./data/FORTE_brain.json', 'r') as f:
    json_data = json.load(f)

filename = './excel_files/sentencepaired_reports.xlsx'
process_excel(filename, json_data)
