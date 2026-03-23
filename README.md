# NLP Sentence Splitter

A sentence splitting project for English and Italian built with character-based neural models, feature engineering, and a small ensemble on top.

## Results

### Neural models

| Model | English Accuracy | English F1 | Italian Accuracy | Italian F1 |
|---|---:|---:|---:|---:|
| CharCNN + features | 0.9651 | 0.9533 | 0.9691 | 0.9579 |
| BiLSTM + features | 0.9795 | 0.9727 | 0.9796 | 0.9728 |
| Meta ensemble | 0.9797 | 0.9731 | 0.9783 | 0.9712 |

### Classical baseline

| Model | English Accuracy | English F1 |
|---|---:|---:|
| Calibrated stacked n-gram model | 0.8779 | 0.9002 |

### NLTK and spaCy baselines

| Baseline | English F1 | Italian F1 |
|---|---:|---:|
| NLTK | 0.9185 | 0.9364 |
| spaCy | 0.6811 | 0.9265 |

## What this project does

This project treats sentence splitting as a binary classification problem over candidate boundaries such as `.`, `?`, `!`, and newline characters.

Each candidate is represented with:

- a centered character window
- handcrafted numeric features
- language-specific preprocessing that removes `<EOS>` from the text seen by the models

The main experiments include:

- a CharCNN with handcrafted features
- a BiLSTM with handcrafted features
- a calibrated n-gram baseline with feature engineering
- a meta-classifier that combines the CharCNN and BiLSTM outputs

The final pipeline supports both English and Italian.

## Setup
> Requires: **Python 3.10** and **CONDA** 

Clone the repo: 

```bash
git clone https://github.com/anilegin/nlp-sentenceSplitter.git
cd nlp-sentenceSplitter
```
Install the dependencies

```bash
conda env create -f environment.yml
conda activate nlp-sentenceSplitter
```

If you want to try the project quickly, you can also use the Colab notebook:

<a href="https://colab.research.google.com/drive/1D17B63ifiHvtW3TXll9UmQQ8mb7aoLgV?usp=sharing" target="_blank">Open in Colab</a>

### Reproducing the experiments

The notebooks are split by experiment:

    notebooks/N-grams_and_feature_engineering.ipynb for classical baselines
    notebooks/CharCNN.ipynb for the CharCNN experiments
    notebooks/BiGRU_BiLSTM.ipynb for recurrent neural models
    main.ipynb for the final ensemble and training pipeline
    evaluate.ipynb for running the saved models on a raw input file

The final models are loaded through inference.py.

In evaluate.ipynb, you only need to provide:

> language, either english or italian

> raw_file_path, the path to a raw text file

Example:

## Project layout

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── english/
│   └── italian/
├── notebooks/
│   ├── CharCNN.ipynb
│   ├── BiGRU_BiLSTM.ipynb
│   ├── N-grams_and_feature_engineering.ipynb
│   └── NLTK_spaCy_split.ipynb
├── results/
│   └── nltk_spacy_results.csv
├── utils/
│   ├── featureExtractor.py
│   ├── preprocessing.py
│   └── text_features.py
├── environment.yml
├── inference.py
├── evaluate.ipynb
├── main.ipynb
└── preprocess.ipynb