# Underrepresented, Overexposed: How Multilingual LLMs Memorized in Low Resourced Languages 

In this work, we introduce the idea that underrepresented languages in multilingual large language models (LLMs) are not simply underfitted, as previous studies suggest, but rather overexposed. While much prior work has argued that LLMs underperform on low-resource languages due to limited training data and NLP resources (such as tokenizers), our preliminary experiments suggest a different perspective.

We begin with the assumption that low-resource languages, such as Kinyarwanda, suffer from limited datasets and NLP tools, combined with complex morphology that existing NLP architectures fail to fully capture. Therefore, in striving to achieve decent performance, these models may sacrifice generalization in favor of memorization. Based on this, we hypothesize that multilingual LLMs have, in fact, overfitted on the limited training data available for low-resource languages.

A key challenge in rigorously testing this hypothesis is that we do not have direct access to the training data used in most LLMs. To navigate this, we leverage NLP intuition: if a model has memorized its training data, this should manifest in specific, detectable behaviors. We propose a study across three languages: one low-resource, one medium-resource, and one high-resource using Named Entity Recognition (NER) as the primary lens of this analysis.

