import json

# Provide the path to your JSON file here
file_path = "/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/final_prompt_ner_clusters_true_en.json"

# Load the JSON from the file
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Calculate repetitions per prompt
repetition_summary = []

for prompt in data:
    prompt_number = prompt['prompt_number']
    total_repetition = 0
    
    for cluster in prompt['ner_clusters']:
        appeared_in = cluster['appeared_in']
        if appeared_in > 1:
            total_repetition += (appeared_in - 1)
    
    repetition_summary.append({
        "prompt_number": prompt_number,
        "total_repetition": total_repetition
    })

with open("repetition_summary_en.json", "w", encoding="utf-8") as f:
    json.dump(repetition_summary, f, indent=4)
