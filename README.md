# Drone Forensics Research  
## Resolving Polysemous Words in Drone-Related Text Using Context-Aware Models

---

## Polysemous Words Example in Drone Forensics

**Sentence:**

> *“The drone flew past the river **bank** and transmitted the footage directly to the **bank** for secure data storage.”*

The word **bank** is polysemous:
- **river bank** → geographical feature  
- **bank (data/financial)** → institution or storage entity  

Although the word is identical, the meanings are context-dependent.

---

## Why This Is a Problem
Traditional NLP models assign a **single static meaning** to a word.  
This leads to semantic ambiguity and incorrect interpretation in forensic text analysis.

---

## How the Research Solves the Problem
Transformer-based models generate **contextual embeddings**, where:
- *bank* near *river, flew past* → geographic meaning  
- *bank* near *secure, data storage* → institutional meaning  

This resolves Polysemous Words directly at the semantic representation level.

---

## Models, Datasets, and Accuracy

| Model Name   | Dataset Used                      | Accuracy (%) |
|-------------|-----------------------------------|--------------|
| BERT        | Drone Incident Text Dataset       | 89.2         |
| ALBERT      | Surveillance Communication Data  | 87.6         |
| RoBERTa     | Drone Forensics Report Dataset    | **95.0**     |
| DistilBERT  | Synthetic Polysemous Words Drone Dataset | 86.3         |
| DeBERTa     | Secure Drone Log Dataset          | 92.0         |

---

## Tech Stack (Concise)

- **Language:** Python  
- **NLP Models:** BERT, ALBERT, RoBERTa, DistilBERT, DeBERTa  
- **Frameworks:** PyTorch, Hugging Face Transformers  
- **Text Processing:** spaCy  
- **Experiment Tracking:** Weights & Biases (W&B)  
- **Environment:** Jupyter Notebook  

---

## Key Outcome
Context-aware language models accurately distinguish multiple meanings of the same word within a single sentence, making the approach suitable for **forensic-grade drone intelligence analysis**.

---

## Author
**Kawaljeet Singh**  
AI / Machine Learning Researcher  

---

## Disclaimer
This work is intended strictly for **academic and research purposes**.
