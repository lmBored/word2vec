# Word2Vec

## Dataset

### Text8 Dataset

**Source**: http://mattmahoney.net/dc/text8.zip

Text8 is a standard dataset derived from the first 100MB of cleaned Wikipedia text (Mahoney, 2006). It was used in the original word2vec experiments.

| Property | Value |
|----------|-------|
| Tokens | 17 million |
| Vocabulary | 60,000 |
| Size | 95.4 MB |
| Content | English Wikipedia |

### Evaluation Datasets

#### WordSim-353: Word relatedness
- 353 word pairs with human similarity ratings (0-10 scale)
- Measures semantic relatedness (similarity + association)
- Classic benchmark

#### SimLex-999: Word similarity
- 999 word pairs rated for genuine similarity
- Distinguishes similarity from association ("coffee" and "cup" are related but not similar)
- More challenging and linguistically principled

#### Google Analogies: Word analogy accuracy
- 19,544 analogy questions across 14 categories
- 5 semantic categories (capital-country, currency, etc.)
- 9 syntactic categories (verb tenses, plurals, etc.)

## Implemented Methods

+ Skip-gram with Negative Sampling (SGNS)
+ Continuous Bag of Words (CBOW) with Negative Sampling

Both with JIT-compiled forward/backward and gradient update

## Usage

### Install requirements

```bash
# Install with uv
uv sync

# Or with pip
pip install -e .
```

### Training

```bash
# CBOW
uv run train_cbow.py

#SGNS
uv run train_sgns.py
```

### Evaluate

```bash
uv run python evaluate.py --model cbow --epochs 3

# To show all available flags
uv run python evaluate.py --help
```

**Success metrics**
1. **Word Relatedness**: Spearman rho > 0.5 (WordSim-353)
2. **Word Similarity**: Spearman rho > 0.3 (SimLex-999)
3. **Word Analogy**: Accuracy > 40% on semantic subset (Text8 baseline)
