import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import os

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize model once
model = SentenceTransformer('all-mpnet-base-v2')

def process_row(row, model):
    gt = str(row['parsed_output'])
    pred = str(row['gt'])
    gt_sents = gt.strip().split('.')
    pred_sents = pred.strip().split('.')
    gt_sents = [sent.strip() for sent in gt_sents if sent.strip()]
    pred_sents = [sent.strip() for sent in pred_sents if sent.strip()]

    if not pred_sents or not gt_sents:
        return []

    # Split into smaller batches to prevent GPU memory overflow
    batch_size = 10  # Adjust as needed
    results = []

    for i in range(0, len(pred_sents), batch_size):
        batch_pred_sents = pred_sents[i:i + batch_size]
        embeddings1 = model.encode(batch_pred_sents, convert_to_tensor=True)
        embeddings2 = model.encode(gt_sents, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)

        for j in range(len(batch_pred_sents)):
            similarity = cosine_scores[j]
            most_similar_sentence_index = similarity.argsort()[-1]
            results.append({
                'gt': batch_pred_sents[j],
                'parsed_output': gt_sents[most_similar_sentence_index],
                'Similarity': float(similarity[most_similar_sentence_index])
            })

    return results

# Read the Excel file with all sheets
xls = pd.ExcelFile('./excel_files/evaluation_examples.xlsx')
sheet_names = xls.sheet_names  # Get all sheet names

with pd.ExcelWriter('./excel_files/sentencepaired_reports.xlsx') as writer:
    for sheet_name in tqdm(sheet_names, desc='Processing Sheets'):
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # Process each row and store the results
        all_results = {}
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Processing Rows in {sheet_name}'):
            id = str(row['id'])
            if id not in all_results:
                all_results[id] = []
            all_results[id].extend(process_row(row, model))

        flattened_results = []
        for key, value in all_results.items():
            for entry in value:
                entry['ID'] = key  # Add the ID to each entry
                flattened_results.append(entry)

        result_df = pd.DataFrame(flattened_results)

        # Write the results to the current sheet in the Excel file, if not empty
        if not result_df.empty:
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            print(f"Sheet '{sheet_name}' is empty and will not be saved.")
