from sentence_transformers import SentenceTransformer, util
import pandas as pd
import ast
from collections import defaultdict

# Load your embed model
model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv('embedded_en.csv')
df['grouped_ner'] = df['grouped_ner'].apply(ast.literal_eval)

final_results = []

for prompt_number, group_df in df.groupby('prompt_number'):
    all_entities = defaultdict(list)  # {PER: [ (cluster_text, generation_number), ...], etc}

    for _, row in group_df.iterrows():
        gen_num = row['generation_number']
        grouped_ner = row['grouped_ner']

        for ent_type, clusters in grouped_ner.items():
            for cluster in clusters:
                text = " ".join(cluster)  # merge cluster into a string
                all_entities[ent_type].append((text, gen_num))

    merged_clusters = defaultdict(list)

    for ent_type, ent_list in all_entities.items():
        texts = [x[0] for x in ent_list]
        gens = [x[1] for x in ent_list]

        # Get embeddings
        embeddings = model.encode(texts, convert_to_tensor=True)

        used = set()
        for i in range(len(texts)):
            if i in used:
                continue
            current_cluster = [(texts[i], gens[i])]
            used.add(i)

            for j in range(i+1, len(texts)):
                if j in used:
                    continue
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                if sim >= 0.7:
                    current_cluster.append((texts[j], gens[j]))
                    used.add(j)

            merged_clusters[ent_type].append(current_cluster)

    # Now for each merged_cluster, count unique generations
    ner_summary = []
    for ent_type, clusters in merged_clusters.items():
        for cluster in clusters:
            unique_gens = set(gen for _, gen in cluster)
            text = " | ".join(sorted(set(text for text, _ in cluster)))
            ner_summary.append({
                "entity_cluster": f"{ent_type}: {text}",
                "appeared_in": len(unique_gens)
            })

    final_results.append({
        "prompt_number": prompt_number,
        "ner_clusters": ner_summary
    })

# Save
import json
with open("final_prompt_ner_clusters_true_en.json", "w") as f:
    json.dump(final_results, f, indent=4)
