# Toxic Comment Classifier (Prototype)

> **Status:** Prototype completed – 2-week project, built as part of a data science training program.

A prototype that detects and classifies toxic comments in English and suggests moderation actions. German data was analyzed in depth, but the prototype focuses on English (see [Why English only?](#why-english-only)). The goal is to support community managers in checking comments on platforms and news sites.

> ⚠️ **Content warning:** The datasets used in this project contain real offensive, obscene and hateful language.

---

## Goals

- Detect toxic comments in **English**; analyze **German** data and document what a German model would require
- Output a **toxicity score (0–1)** plus categories: insult, threat, obscene, identity attack, sexual explicit
- Suggest a **moderation action** for each comment, from "no action" to "hide and escalate to a moderator"
- Provide a **user interface** where users can type a comment and immediately see the result
- **Critically evaluate** the prototype: where is it reliable, where does it fail, and whom might it treat unfairly?

**Guiding principle:** The software recommends, a human decides. No action is ever executed automatically.

---

## Approach

| Step | Method |
| --- | --- |
| Baseline | TF-IDF (word and character n-grams) + logistic regression, with a learning curve from 10k to 1.8M comments |
| Category models | Word TF-IDF with word pairs, one logistic regression per category, trained on 521,155 English comments |
| Thresholds | Best F1 per category, selected on the validation data |
| Explanations | Word weights of the logistic regression show which words drive the score |
| Recommendations | Rule-based `recommend()` function mapping scores to action levels 0–5 |
| Interface | Streamlit app with single-comment input; actions and CSV upload shown as disabled previews |
| Critical evaluation | Bias test with neutral template sentences, error analysis |

The prototype deliberately uses classical machine learning and runs entirely on a standard laptop (MacBook M1, 8 GB RAM).
Training, validation and test data are strictly separated; the test split was used exactly once for the final evaluation.

---

## Key findings from the data analysis

- **Different base rates:** 8 % toxic in Civil Comments, 4.7 % in RP-Crowd, 35 % in GermEval 2021. Results are therefore reported per source, and no single threshold fits all sources.
- **Moderation ≠ toxicity:** Only 21 % of the comments rejected by RP moderators were also rejected by the crowd. Crowd votes were used as labels.
- **Data leakage:** 658 English training texts also appeared in the evaluation data and were removed from the training data.
- **Confounder:** Placeholders like @USER appeared in 1,000 of 2,902 training comments. It would have let a model recognize the source instead of toxicity, so it was removed.
- **Rare categories:** Threats are rare in English (0.26 %) and almost absent in German (22 in the training data).
- **Identity-term bias:** Group terms appear far more often in toxic comments (English about 3x, German up to 9x the base rate).

---

## Why English only?

German turned out to be much harder than expected (PR-AUC 0.25 vs. 0.66 for English with the same baseline). The decision is based on measurements:

| Problem | Evidence | What would be needed |
| --- | --- | --- |
| Too little data | 4,200 toxic German training comments vs. 60,000 English; only 22 threats | Much more labeled data |
| Missing categories | Language specifics | Models that understand subwords (transformers) |
| No transfer | TF-IDF cannot transfer knowledge from English to German | A multilingual model and a GPU |
| Stronger bias | Group terms up to 9x overrepresented in toxic comments | Targeted bias checks and more balanced data |

---

## Results

### Final evaluation (English test split, used once)

| Label | PR-AUC (validation) | PR-AUC (test) | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| toxicity | 0.677 | 0.673 | 0.603 | 0.645 | 0.624 |
| insult | 0.675 | 0.669 | 0.597 | 0.657 | 0.625 |
| obscene | 0.506 | 0.477 | 0.449 | 0.602 | 0.514 |
| identity_attack | 0.295 | 0.299 | 0.330 | 0.482 | 0.392 |
| sexual_explicit | 0.214 | 0.214 | 0.333 | 0.258 | 0.291 |
| threat | 0.175 | 0.164 | 0.212 | 0.262 | 0.234 |

Precision, recall and F1 at the thresholds chosen on the validation data. Test and validation results match closely, and there is no meaningful overfitting (training–validation ROC-AUC gap +0.007 to +0.022).
ROC-AUC is between 0.94 and 0.97 for all labels and is misleadingly optimistic for rare classes; PR-AUC is the main metric.

### Recommendations on the test split

| Level | Action | Comments | Truly toxic |
| --- | --- | --- | --- |
| 0 | No action | 83,515 (86 %) | 1.9 % |
| 1 | Monitor | 5,227 | 21 % |
| 2 | Manual review | 5,356 | 53 % |
| 3 | Hide until review | 1,012 | 37 % |
| 4 | Delete | 1,936 | 90 % |
| 5 | Hide and escalate to a moderator | 274 | 38 % |

- "Delete" is justified in 9 out of 10 cases.
- Levels 2–5 cover about 9 % of all comments and contain about 65 % of all toxic comments.
- Level 3 is less precise than level 2 because identity-attack signals are biased (see below).
- Level 5 (possible threat) escalates instead of deleting, because the threat model is the weakest.

### Bias test (84 neutral template sentences)

Six harmless templates ("My neighbor is a ___ man.", "She is a ___ doctor.", …) with 14 groups. Every alarm is a false alarm; "tall" is a control word without an identity reference.

| Group | Mean toxicity | Hidden |
| --- | --- | --- |
| gay | 0.82 | 6 of 6 |
| Black | 0.71 | 6 of 6 |
| Muslim | 0.57 | 6 of 6 |
| white | 0.53 | 6 of 6 |
| straight, American, Asian, tall | 0.06–0.08 | 0 of 6 |

The same harmless sentence is treated completely differently depending on one word.

**Design decision:** Comments with identity-attack signals are hidden first and reviewed afterwards, because the harm of a visible attack cannot be undone, while a hidden comment can be restored. The bias test quantifies the cost of this decision. A tool like this must only be used with human review.

### Main limitations

The model scores words, not meaning.

Sarcasm and obfuscated spellings are also missed. More training data does not solve this: the learning curve flattens after about 300,000 comments.

---

## Outlook

Each possible next step follows from a limitation found in this prototype:

- **Understand meaning:** a transformer model (e.g. mmBERT), trained on a GPU (e.g. Google Colab). It would likely reduce errors with criticism of things, implicit threats and compounds, but would not remove the identity-term bias automatically, because it learns from the same labels.
- **Uncertain cases:** a language model as a second opinion, with a human still deciding.
- **Bias:** more balanced training data and the template test as a standard check for every new model.
- **German:** more and more consistently labeled data (see [Why English only?](#why-english-only)).
- **Product:** CSV upload, real moderation actions and platform integration (already shown as disabled previews in the app).
- **Conversation quality:** also recognizing spam, constructive contributions and factual criticism.

---

## Components and licenses

| Component | Purpose | License |
| --- | --- | --- |
| Civil Comments | Training and evaluation data (EN) | CC0 |
| RP-Mod & RP-Crowd | Analysis and baseline (DE) | CC BY-NC-SA 4.0 |
| GermEval 2021 | Analysis and baseline (DE) | Not specified, educational use only |
| pandas, NumPy, scikit-learn | Data processing and models | BSD-3-Clause |
| matplotlib, seaborn | Visualizations | Matplotlib License, BSD-3-Clause |
| Streamlit | User interface | Apache 2.0 |
| Hugging Face `datasets` | Data download | Apache 2.0 |

---

## Disclaimer

This is a learning prototype. Its predictions can be wrong, especially for irony, sarcasm, quotes, implicit threats and slang, and they are biased against certain identity terms. It is not intended for automated moderation decisions. Using it on real platform comments would require a data protection (GDPR) assessment.

## License

No license has been chosen for the code. All rights are reserved.
