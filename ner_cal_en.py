import json
import pandas as pd
import os
from transformers import pipeline
import ast  # Add this for safe string-to-dict conversion

FOLDER_PATH = "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/Row_Resourced/en_data" 
OUTPUT_CSV = "En_NER_output.csv"

ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

data_rows = []

for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(".json"):
        json_path = os.path.join(FOLDER_PATH, filename)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            stories = json.load(f)
            
            for story in stories:
                story_id = story.get("story_id", None)
                text = story.get("text", "")
                
                ner_results = ner_pipeline(text)
                
                grouped_entities = {}
                for ent in ner_results:
                    label = ent['entity_group']
                    entity = ent['word']
                    if label not in grouped_entities:
                        grouped_entities[label] = []
                    if entity not in grouped_entities[label]:
                        grouped_entities[label].append(entity)
                
                total_entities = sum(len(v) for v in grouped_entities.values())
                
                data_rows.append({
                    'json_file': filename,
                    'story_id': story_id,
                    'ner_tagging': str(grouped_entities),  # Convert dict to string explicitly
                    'number_of_ner_tags': total_entities
                })

# Create DataFrame and save
df = pd.DataFrame(data_rows)
output_path = os.path.join(FOLDER_PATH, OUTPUT_CSV)
df.to_csv(output_path, index=False)

print(f"NER results saved to: {output_path}")