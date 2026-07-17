# Trendyol E-Commerce Search Relevance AI System
## Technical Design Document

---

# Objective

Build a **state-of-the-art Search Relevance AI System** capable of achieving **Top-1 Kaggle performance (Macro F1 > 0.98)** in the Trendyol E-Commerce Search Relevance Competition.

The solution **must not** be implemented as a simple Transformer classifier.

Instead, design and implement a **production-grade industrial search relevance engine**, inspired by large-scale e-commerce search systems such as Amazon, Alibaba, Google Shopping and Trendyol.

The entire system should prioritize:

- Highest possible Macro F1
- Explainability
- Scalability
- Modularity
- Experimentation
- Reproducibility
- Production-readiness

The architecture should be flexible enough to evolve during the competition without requiring major rewrites.

---

# Overall System Architecture

```
                         Raw CSV Files
                               │
                               ▼
                        Data Loading Layer
                               │
                               ▼
                     Data Validation Layer
                               │
                               ▼
                     Data Cleaning Pipeline
                               │
                               ▼
                  Product Normalization Layer
                               │
                               ▼
                     Feature Engineering
                               │
                               ▼
                  Negative Sample Generation
                               │
                               ▼
                      Dataset Construction
                               │
                               ▼
                 Train / Validation Split
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
         Embedding Models             Cross Encoder
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                       Feature Fusion Layer
                               ▼
                         Meta Classifier
                               ▼
                    Threshold Optimization
                               ▼
                         Inference Engine
                               ▼
                      Submission Generator
```

---

# Folder Structure

```
project/

configs/

data/
    raw/
    processed/
    cache/

features/

negative_samples/

embeddings/

models/
    cross_encoder/
    embedding/
    meta/

training/

validation/

inference/

submission/

experiments/

reports/

logs/

utils/

tests/

notebooks/
```

The entire project must be configuration-driven.

Never hardcode paths or parameters.

---

# Data Loading

Load the following files:

- items.csv
- terms.csv
- training_pairs.csv
- submission_pairs.csv
- sample_submission.csv

Merge datasets into unified structures.

Every training sample should contain:

```
Query

Title

Category

Brand

Gender

Age Group

Attributes

Label
```

No additional lookup should be necessary during training.

---

# Data Validation

Before any processing:

Validate

- Missing IDs
- Duplicate IDs
- Invalid UTF-8
- Empty titles
- Missing queries
- Missing categories
- Broken attributes
- Incorrect delimiters

Generate validation reports automatically.

---

# Product Normalization

Raw product information should never be passed directly into models.

Convert every product into structured text.

Example

```
Title:
Erkek Oversize Pamuklu Tişört

Category:
Erkek > Giyim > Tişört

Brand:
Defacto

Gender:
Erkek

Age Group:
Yetişkin

Attributes:
Color: Black
Material: Cotton
Pattern: Plain
Collection: Basic
```

Always use identical formatting.

---

# Query Normalization

Normalize every query.

Perform:

- Unicode normalization
- Turkish character normalization
- Lowercase conversion
- Duplicate whitespace removal
- Punctuation normalization
- Symbol cleanup

Preserve semantic meaning.

---

# Attribute Parsing

Attributes currently exist as one long string.

Convert

```
material: cotton,
color: black,
pattern: plain
```

into structured dictionaries.

Example

```python
{
    "material": "cotton",
    "color": "black",
    "pattern": "plain"
}
```

Then reconstruct clean textual representation.

---

# Category Engineering

Split category hierarchy.

Example

```
Accessories/Bag/Shoulder Bag
```

becomes

```
Level 1
Accessories

Level 2
Bag

Level 3
Shoulder Bag
```

Store every level independently.

---

# Brand Normalization

Normalize brands.

Examples

```
nike

Nike

NIKE
```

↓

```
nike
```

Remove unnecessary whitespace.

Fix duplicated brand names.

---

# Text Cleaning Pipeline

Implement:

- Unicode normalization
- Turkish normalization
- HTML cleanup
- Multiple whitespace removal
- Attribute normalization
- Category normalization
- Brand normalization
- Gender normalization
- Age normalization

Every step should be modular.

---

# Feature Engineering

Generate structured features.

Examples

- Query length
- Product title length
- Number of attributes
- Category depth
- Brand existence
- Attribute count
- Token count
- Exact token overlap
- Query title overlap
- Category overlap
- Brand overlap

Every feature should be independently switchable.

---

# Negative Sample Generation

This module is the most important part of the competition.

Training data contains only positive samples.

Generate negatives using multiple independent strategies.

## Strategy 1

Random negatives

Random products.

Different category.

---

## Strategy 2

Category negatives

Same category.

Different brand.

---

## Strategy 3

Brand negatives

Same brand.

Different product.

---

## Strategy 4

Hard negatives

Embedding nearest neighbors.

Wrong products.

---

## Strategy 5

Adversarial negatives

Almost identical products.

Different

- color
- size
- SKU
- model

---

## Strategy 6

Semantic negatives

Nearest embedding neighbors.

Wrong relevance.

---

Negative sampling ratios must be configurable.

Example

```
Easy

20%

Medium

30%

Hard

30%

Adversarial

20%
```

---

# Dataset Builder

Create balanced datasets.

Example

```
Positive

250,000

Negative

750,000
```

Support configurable ratios.

---

# Validation Split

Never use random split.

Use Group Split.

The same query must never appear in both training and validation.

Prevent data leakage.

---

# Embedding Models

Support multiple embedding models.

Examples

- BGE-M3
- E5 Multilingual
- Jina Embeddings
- Sentence Transformers

Embeddings must be cached.

Never regenerate embeddings unnecessarily.

---

# Cross Encoder

Main relevance model.

Candidate models

- XLM-R Large
- BERTurk
- ModernBERT
- Qwen Embedding Cross Encoder

Input format

```
Query

[SEP]

Product
```

Output

```
Probability

0.0

↓

1.0
```

---

# Similarity Features

Generate

- Cosine similarity
- Dot product
- Euclidean distance
- Manhattan distance
- Jaccard similarity
- BM25 score
- TF-IDF similarity
- Levenshtein distance

---

# Meta Features

Generate additional handcrafted features.

Examples

Brand Match

Category Match

Gender Match

Age Match

Color Match

Material Match

Pattern Match

Collection Match

Season Match

Exact Match

Partial Match

Token Overlap

Attribute Overlap

Title Similarity

Query Similarity

Embedding Distance

Category Distance

---

# Feature Fusion

Never rely solely on Transformer outputs.

Combine

Cross Encoder Score

Embedding Similarity

Handcrafted Features

using

- LightGBM
- XGBoost
- CatBoost

Support interchangeable meta-models.

---

# Training Pipeline

Support

- Mixed Precision
- Gradient Accumulation
- Gradient Clipping
- Automatic Resume
- Checkpoint Saving
- Early Stopping
- Learning Rate Scheduler
- Seed Control

Every experiment must be reproducible.

---

# Hyperparameter Optimization

Implement Optuna.

Search

- Learning Rate
- Batch Size
- Warmup Ratio
- Weight Decay
- Dropout
- Scheduler
- Threshold

Automatically store best configuration.

---

# Threshold Optimization

Macro F1 depends heavily on threshold.

Search optimal threshold automatically.

Example

```
0.50

↓

0.58

↓

0.61

↓

0.64
```

Store best threshold.

---

# Experiment Tracking

Every experiment should automatically save

- Configuration
- Validation Score
- Macro F1
- Precision
- Recall
- Threshold
- Runtime
- Model Path
- Git Commit
- Date
- Notes

No experiment should ever be lost.

---

# Inference Pipeline

```
Submission Pair

↓

Query

+

Product

↓

Embedding

↓

Cross Encoder

↓

Meta Features

↓

Meta Model

↓

Probability

↓

Threshold

↓

Prediction
```

---

# Submission Generator

Generate

submission.csv

Validate

- Duplicate IDs
- Missing IDs
- Wrong ordering
- Wrong labels

Automatically compare with sample submission.

---

# Error Analysis

Generate reports.

Examples

Most common

- False Positives
- False Negatives
- Worst Queries
- Worst Categories
- Worst Brands
- Worst Colors
- Worst Attributes

Visualize confusion patterns.

---

# Explainability

Generate explanations.

Example

```
Relevant because

✓ Brand matched

✓ Category matched

✓ Material matched

✓ High semantic similarity

✓ Cross Encoder confidence 0.98
```

Explainability should work automatically.

---

# Performance Optimization

Implement

- Embedding Cache
- Feature Cache
- Parsed Attribute Cache
- Tokenization Cache

Support multiprocessing wherever beneficial.

Avoid recomputation.

---

# Logging

Every module should produce structured logs.

Example

```
Dataset Loaded

Negative Sampling Started

Training Started

Validation Finished

Submission Generated
```

Support log levels.

---

# Configuration System

All parameters must be configurable.

Example

```yaml
negative_sampling:
    easy_ratio: 0.2
    medium_ratio: 0.3
    hard_ratio: 0.3
    adversarial_ratio: 0.2

training:
    batch_size: 32
    epochs: 5
    learning_rate: 2e-5
```

Never hardcode hyperparameters.

---

# Code Quality

Use

- Type Hints
- Dataclasses
- Pydantic
- Logging
- Unit Tests

Avoid duplicated code.

Every module must be independently testable.

---

# Future Extensions

Design the system to easily support

- Elasticsearch Retrieval
- Hybrid Search
- Vector Search
- FAISS
- Milvus
- Qdrant
- Multi-stage Ranking
- LLM Explainability
- Online Inference API

No future extension should require major architectural changes.

---

# Final Goal

The project must resemble a **production-grade industrial search relevance platform**, not a simple Kaggle notebook.

The architecture should support:

- Continuous experimentation
- Easy model replacement
- Modular feature engineering
- Scalable inference
- Explainable predictions
- Maximum possible Macro F1

The primary objective is to build a system capable of achieving **Top-1 Kaggle performance (>0.98 Macro F1)** while remaining clean, extensible, reproducible and suitable for real-world deployment.