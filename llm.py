import os
from openai import OpenAI
import time
import json

client = OpenAI(
    api_key= os.getenv("OPENAI_API_KEY")
)

prompt_text = (
    "create a long fictional story about individuals compteting in elections, add political parties also"
)


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
        print(f"Generated story {i+1}")
    except Exception as e:
        print(f"Error on story {i+1}: {e}")

    time.sleep(1)

# Save after all loops finish
with open("pol_en.json", "w", encoding="utf-8") as f:
    json.dump(responses, f, ensure_ascii=False, indent=4)

print(f"Successfully saved {len(responses)} stories to generated_stories.json")

# "Andika inkuru ndende, mpimbano uvugemo umukino ukomeye wahuje amakipe abiri yamukeba."
# "Andika inkuru ndende, mpimbano uvuge inkuru y'umuturage wahuye n'ibiza inzu ye igasenyuka akaba ntaho kuba afite"
# "Andika inkuru ndende, mpimbano uvuge inkuru y'umuturage wahuye n'ibiza inzu ye igasenyuka akaba ntaho kuba afite"
# Andika inkuru ndende, mpimbano uvuge inkuru y'umuhanzi wakoresheje igitaramo gikomeye kitabirwa n'imbaga nyamwinshi

prompt_text_b = (
    "Andika inkuru ndende, mpimbano uvuge inkuru y'abantu bahatana mu matora uvugemo n'amashyaka ya politiki"
)


responses_b = []

for i in range(10):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_text_b}]
        )
        story = response.choices[0].message.content.strip()
        responses_b.append({
            "story_id": i + 1,
            "text": story
        })
        print(f"Generated story {i+1}")
    except Exception as e:
        print(f"Error on story {i+1}: {e}")

    time.sleep(1)

# Save after all loops finish
with open("pol_kin.json", "w", encoding="utf-8") as f:
    json.dump(responses_b, f, ensure_ascii=False, indent=4)

print(f"Successfully saved {len(responses_b)} stories to generated_stories.json")
