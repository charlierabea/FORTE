import pandas as pd
from sentence_transformers import SentenceTransformer, util

def process_row(row):
    gt = str(row['parsed_output'])
    pred = str(row['gt'])
    gt_sents = gt.strip().split('.')
    pred_sents = pred.strip().split('.')
    gt_sents = [sent.strip() for sent in gt_sents if sent.strip()]
    pred_sents = [sent.strip() for sent in pred_sents if sent.strip()]

    if not pred_sents or not gt_sents:
        return []

    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings1 = model.encode(pred_sents, convert_to_tensor=True)
    embeddings2 = model.encode(gt_sents, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    results = []
    for i in range(len(pred_sents)):
        similarity = cosine_scores[i]
        most_similar_sentence_index = similarity.argsort()[-1]
        results.append({
            'gt': pred_sents[i],
            'parsed_output': gt_sents[most_similar_sentence_index],
            'Similarity': float(similarity[most_similar_sentence_index])
        })

    return results

# Read the Excel file with all sheets
xls = pd.ExcelFile('./excel_files/evaluation_examples.xlsx')
sheet_names = xls.sheet_names  # Get all sheet names

with pd.ExcelWriter('./excel_files/sentencepaired_reports.xlsx') as writer:
    for sheet_name in sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # Process each row and store the results
        all_results = {}
        for _, row in df.iterrows():
            id = str(row['id'])
            if id not in all_results:
                all_results[id] = []
            all_results[id].extend(process_row(row))

        flattened_results = []
        for key, value in all_results.items():
            for entry in value:
                entry['ID'] = key  # Add the ID to each entry
                flattened_results.append(entry)

        result_df = pd.DataFrame(flattened_results)

        # Write the results to the current sheet in the Excel file
        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
