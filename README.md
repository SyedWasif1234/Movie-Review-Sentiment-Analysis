# Movie Review Sentiment Analysis: LSTM Baseline vs Fine-Tuned DistilBERT

This README explains **every line** of the transformer part of the project, the theory behind it, likely interview questions, and a method for writing this kind of code from documentation instead of asking an LLM.

---

## 1. Project Overview

| Item | Detail |
|---|---|
| Task | Binary sentiment classification (positive / negative) of movie reviews |
| Dataset | IMDB-style CSV with a `review` column and a `sentiment` column |
| Baseline | RNN / LSTM with lowercasing, tokenization, stemming, vectorization |
| Improved model | Fine-tuned `distilbert-base-uncased` (transfer learning) |
| Metrics | Accuracy and F1 on the same held-out test set for both models |

### Pipeline

```
Raw reviews -> minimal cleaning -> label encoding -> train/val/test split
   -> Hugging Face Dataset -> tokenizer -> pretrained model + classifier head
   -> fine-tune with Trainer -> evaluate on test set -> save -> predict
```

### RNN vs Transformer preprocessing

| Step | RNN | Transformer |
|---|---|---|
| Lowercasing | Manual | Done by the tokenizer (uncased model) |
| Tokenization | Word-level, manual | Subword (WordPiece), by the model's tokenizer |
| Stemming | Yes | **No** (creates tokens like "movi" the model never saw in pretraining) |
| Stopword removal | Often | **No** (words like "not" flip sentiment) |
| Vectorization | Embedding layer trained from scratch | Pretrained embeddings inside the model |
| Label encoding | Yes | Yes |

---

## 2. Full Code With Line-by-Line Explanation

### 2.1 Installation

```bash
pip install transformers datasets evaluate accelerate scikit-learn
```
