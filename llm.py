import os
import time
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Read prompts from a text file
with open("kinyarwanda_prompts.txt", "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

for idx, prompt_text in enumerate(prompts, start=1):
    responses = []

    for i in range(10):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_text}]
            )
            story = response.choices[0].message.content.strip()
            responses.append({
                "story_id": i + 1,
                "text": story
            })
            print(f"Prompt {idx:02}, story {i+1} generated.")
        except Exception as e:
            print(f"Error on prompt {idx:02}, story {i+1}: {e}")

        time.sleep(1)

    filename = f"{idx:02}_answer.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(responses)} stories to {filename}")
