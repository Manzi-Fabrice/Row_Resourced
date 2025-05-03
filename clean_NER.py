import pandas as pd
import ast
import re

# Load CSV
df = pd.read_csv("/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/Row_Resourced/en_data/En_NER_output.csv")

valid_tags = {'PER', 'ORG', 'LOC', 'DATE'}

# Explicit cleaning function
def clean_ner_entry(entry):
    final_entries = []
    for item in entry:
        if '#' in item:
            continue
        cleaned_item = re.sub(r'\s+', '', item.strip())
        if len(cleaned_item) > 2:
            final_entries.append(item.strip())
    return final_entries

# Robust processing function with explicit checking
def process_ner(ner_str):
    try:
        ner_dict = ast.literal_eval(ner_str)
    except Exception as e:
        print(f"Error parsing entry: {ner_str}, {e}")
        return {}, 0

    cleaned_dict = {}
    total_count = 0

    for tag in valid_tags:
        if tag in ner_dict:
            items = ner_dict[tag]
            cleaned_items = clean_ner_entry(items)
            if cleaned_items:
                cleaned_dict[tag] = cleaned_items
                total_count += len(cleaned_items)

    return cleaned_dict, total_count

# Apply cleaning explicitly and print debug statements
def debug_and_process(row):
    cleaned, count = process_ner(row['ner_tagging'])
    print(f"Original: {row['ner_tagging']}")
    print(f"Cleaned: {cleaned}, Count: {count}\n")
    return pd.Series([cleaned, count])

# Apply processing and get cleaned data
df[['cleaned_ner', 'cleaned_ner_count']] = df.apply(debug_and_process, axis=1)

# Replace the original ner_tagging column with the cleaned data
df['ner_tagging'] = df['cleaned_ner'].astype(str)  # or keep as dict if preferred

# Save to CSV
df.to_csv("/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/Row_Resourced/en_data/cleaned_ner_output.csv", index=False)