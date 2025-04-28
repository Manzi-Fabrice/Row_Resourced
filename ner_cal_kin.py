import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os  # To get just the file name from the path

# Load the NER model
model_name = 'mbeukman/xlm-roberta-base-finetuned-ner-kinyarwanda'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Paths to your datasets
datasets = [
    "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/sport_kin.json",
    "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/showbie_kin.json",
    "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/sad_scandal_kin.json",
    "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/politic_kin.json",
    "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/pol_kin.json",
    "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/nature_kin.json"
]

# Empty list to collect rows
data_rows = []

# Go through each dataset
for dataset_path in datasets:
    dataset_name = os.path.basename(dataset_path)  # Get only the file name e.g., 'nature_kin.json'
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        stories = json.load(f)
        
        for story in stories:
            story_id = story['story_id']
            text = story['text']
            
            ner_results = ner_pipeline(text)
            
            # Group NER results
            grouped_entities = {}
            for r in ner_results:
                label = r['entity_group']
                entity = r['word']
                if label not in grouped_entities:
                    grouped_entities[label] = []
                if entity not in grouped_entities[label]:  # avoid duplicates
                    grouped_entities[label].append(entity)
            
            total_entities = sum(len(v) for v in grouped_entities.values())
            
            # Add row to our data
            data_rows.append({
                'dataset_name': dataset_name,
                'dataset_number': story_id,
                'ner_tagging': grouped_entities,
                'number_of_ner_tags': total_entities
            })

# Create a dataframe
df = pd.DataFrame(data_rows)

# Save to CSV
df.to_csv("/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/ner_output.csv", index=False)

print("Done! CSV created")
