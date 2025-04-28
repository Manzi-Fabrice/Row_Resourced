# Underrepresented Means Overexposed: An Exploration of Memorization in Multilingual LLMs

In this work, we introduce the idea that underrepresented languages in multilingual large language models (LLMs) are not simply underfitted, as previous studies suggest, but rather overexposed. While much prior work has argued that LLMs underperform on low-resource languages due to limited training data and NLP resources (such as tokenizers), our preliminary experiments suggest a different perspective.

We begin with the assumption that low-resource languages, such as Kinyarwanda, suffer from limited datasets and NLP tools, combined with complex morphology that existing NLP architectures fail to fully capture. Therefore, in striving to achieve decent performance, these models may sacrifice generalization in favor of memorization. Based on this, we hypothesize that multilingual LLMs have, in fact, overfitted on the limited training data available for low-resource languages.

A key challenge in rigorously testing this hypothesis is that we do not have direct access to the training data used in most LLMs. To navigate this, we leverage NLP intuition: if a model has memorized its training data, this should manifest in specific, detectable behaviors. We propose a study across three languages: one low-resource, one medium-resource, and one high-resource using Named Entity Recognition (NER) as the primary lens of analysis.

Specifically, we propose the following experimental designs:

## NER Density Comparison

We compare the density of named entities across the three languages. Our hypothesis is that a higher NER density in the low-resource language, relative to the high-resource language, may suggest memorization, since low-resource languages logically should not support stronger generalization than high-resource ones.

## NER Consistency in Repeated Prompts

We test the consistency of NER outputs across multiple generations of the same prompt (e.g., asking the model the same question 10 times). Consistency patterns across low-, medium-, and high-resource languages may reveal the extent of memorization.

## NER Generalization Across Diverse Prompts

We generate a variety of vague, generic prompts, such as “Write a story about a match between two rival teams,” and translate them into each target language using native speaker translators. We generate 10 independent model completions per prompt and analyze the NER distribution. If NER entities vary widely across prompts in high-resource languages but remain narrowly focused or "fixed" in low-resource languages, this would further suggest memorization.

---

# Preliminary Study (Phase 01)

In our initial experiment, we generated 60 completions (6 prompts × 10 completions) in Kinyarwanda (low-resource) and English (high-resource) and analyzed their NER profiles using Hugging Face models.

Contrary to our initial hypothesis, English exhibited a higher NER density than Kinyarwanda. However, it is important to note that the Kinyarwanda NER model’s performance is estimated at 70–80% accuracy, meaning some named entities may have been missed or misclassified.

Despite this, several interesting patterns emerged:

## Use of Real Locations

In Kinyarwanda responses, approximately 80% of the places mentioned were real-world locations (verified via Google Maps), compared to only about 20% in English. This suggests that the model may be drawing heavily from memorized real-world knowledge in Kinyarwanda.

## Full and Realistic Personal Names

The Kinyarwanda outputs often included full names (two to three names) that sounded realistic, whereas the English outputs predominantly used generic names like "Alex," "Lily," and "Bob."

## Domain-Specific Realism

Native speakers' evaluation found that team names generated in Kinyarwanda were actually popular teams in Rwanda, whereas English outputs often invented fictional team names, suggesting stronger creative generalization in English.

---

These findings hint at possible memorization biases in the model's handling of Kinyarwanda compared to English, sparking deeper inquiry into the nature of "real vs. fabricated" entities.
