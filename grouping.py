from sentence_transformers import SentenceTransformer, util
import ast
import re
import pandas as pd

# Load model once
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def group_similar_entities(entities, threshold=0.7):
    if not entities:
        return []
    
    embeddings = model.encode(entities)
    cos_sim_matrix = util.cos_sim(embeddings, embeddings)

    groups = []
    used = set()

    for i in range(len(entities)):
        if i in used:
            continue
        group = [entities[i]]
        used.add(i)
        for j in range(i+1, len(entities)):
            if j not in used and cos_sim_matrix[i][j] > threshold:
                group.append(entities[j])
                used.add(j)
        groups.append(group)
    
    return groups

def process_single_prompt(ner_string):
    try:
        ner_dict = ast.literal_eval(ner_string)
    except:
        return {}

    grouped_ner = {}

    for tag_type, entities in ner_dict.items():
        clean_entities = [e for e in entities if e.strip()]
        grouped = group_similar_entities(clean_entities)
        grouped_ner[tag_type] = grouped

    return grouped_ner

# Load your CSV
input_path = "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/ner_en_output.csv"
output_path = "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/embedded_en.csv"

df = pd.read_csv(input_path)

# Prepare list for final results
results = []

# Process in blocks of 10
for dataset_name in df['dataset_name'].unique():
    df_subset = df[df['dataset_name'] == dataset_name]

    for prompt_num in df_subset['dataset_number'].unique():
        df_prompt = df_subset[df_subset['dataset_number'] == prompt_num]

        # Only pick the FIRST sample for now (you said first one)
        if len(df_prompt) == 0:
            continue
        
        row = df_prompt.iloc[0]  # First run
        grouped_ner = process_single_prompt(row['ner_tagging'])

        results.append({
            'dataset_name': row['dataset_name'],
            'dataset_number': row['dataset_number'],
            'grouped_ner': grouped_ner
        })

# Save to new CSV
df_out = pd.DataFrame(results)
df_out.to_csv(output_path, index=False)

print(f"Done! Saved embedded Kinyarwanda NER to {output_path}")