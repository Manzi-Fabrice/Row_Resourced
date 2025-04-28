import json
import pandas as pd
import os
from transformers import pipeline

# Load Hugging Face NER pipeline
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",  # English NER model
    aggregation_strategy="simple"
)

# Paths to your English datasets
datasets = [
    "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/sport_en.json",
    "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/showbie_en.json",
    "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/sad_scandal_en.json",
    "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/politic_en.json",
    "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/pol_en.json",
    "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/nature_en.json"
]

# Empty list to collect rows
data_rows = []

# Go through each dataset
for dataset_path in datasets:
    dataset_name = os.path.basename(dataset_path)
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        stories = json.load(f)
        
        for story in stories:
            story_id = story['story_id']
            text = story['text']
            
            # Run NER
            ner_results = ner_pipeline(text)
            
            # Group entities by label
            grouped_entities = {}
            for ent in ner_results:
                label = ent['entity_group']
                entity = ent['word']
                if label not in grouped_entities:
                    grouped_entities[label] = []
                if entity not in grouped_entities[label]:  # avoid duplicates
                    grouped_entities[label].append(entity)
            
            total_entities = sum(len(v) for v in grouped_entities.values())
            
            # Add row
            data_rows.append({
                'dataset_name': dataset_name,
                'dataset_number': story_id,
                'ner_tagging': grouped_entities,
                'number_of_ner_tags': total_entities
            })

# Create DataFrame
df = pd.DataFrame(data_rows)

# Save to CSV
df.to_csv("/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/ner_en_output.csv", index=False)

print("✅ English NER output saved to ner_en_output.csv")
