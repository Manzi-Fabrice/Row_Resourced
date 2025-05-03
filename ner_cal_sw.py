import json
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


FOLDER_PATH = "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/sw_data"  
OUTPUT_CSV = "ner_swahili_output.csv"


# Load the Swahili NER model
model_name = "mbeukman/xlm-roberta-base-finetuned-ner-swahili"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

data_rows = []

# Loop over all .json files in the folder
for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(".json"):
        json_path = os.path.join(FOLDER_PATH, filename)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            stories = json.load(f)

            for story in stories:
                story_id = story.get("story_id", None)
                text = story.get("text", "")
                
                # Run NER
                ner_results = ner_pipeline(text)
                
                # Organize entities
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
                    'ner_tagging': grouped_entities,
                    'number_of_ner_tags': total_entities
                })

# Convert to CSV
df = pd.DataFrame(data_rows)
output_path = os.path.join(FOLDER_PATH, OUTPUT_CSV)
df.to_csv(output_path, index=False)

print(f"NER results saved to: {output_path}")
