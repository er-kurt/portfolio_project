# HEADLINE

## **Auflistungen**
- `#` für Überschriften
- `##` für 2. Überschrift / ### für 3. Überschrift usw.
- `*` für *kursiv*
- `**` für **fett**
- `***` für kursiv und fett
- shift + cmd + v zum Markdown Preview
- `[Linktext](https://github.com)`
- `![Alt-Text](bild.png)`
- `~~` für durchgestrichen ~~durchgestrichen~~
- Tabellen erstellen mit (opt+7) | und -
- cmd + k, dann v öffnet preview neben editor
- [x] erledigt
- [ ] noch offen

> Zitat
>
> Toxic Comments Classification

A data science project that classifies comments from different data sources as toxic or non-toxic (and optionally into toxicity subcategories). The final result is a Streamlit app with a text box where users can enter a comment and instantly see the model's prediction.

Status: 🚧 Work in progress – project setup phase

Table of Contents

Project Goal

Data Sources

Project Structure

Approach

Installation

Usage

Streamlit App

Results

Roadmap

Changelog

Project Goal


Collect and combine comment data from multiple sources
Clean, explore and preprocess the text data
Train and evaluate classification models for toxic language
Deploy the best model in an interactive Streamlit app

Project Structure
tbd

Approach
Data collection – gather and merge comments from the selected sources
EDA – class distribution, comment length, frequent terms, label imbalance
Preprocessing – cleaning, normalization, tokenization
Modeling – baseline (e.g. TF-IDF + Logistic Regression), then more advanced models (e.g. transformer-based)
Evaluation – precision, recall, F1-score, confusion matrix
Deployment – Streamlit app for live predictions
