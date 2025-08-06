# Datathon 2025 – Team 1: Antisemitism Detection

This repository documents our participation in Challenge #1 and #2 of the Datathon 2025. Our goal was to collect, annotate, and analyze antisemitic content on Twitter/X using the IHRA Working Definition of Antisemitism. Below we explain the methodology, tools used, and project structure.

## 📌 Project Objectives

- Colect Twitter posts (after Oct 7, 2023) containing antisemitic narratives.
- Annotate tweets based on the IHRA Working Definition of Antisemitism.
- Optionally build a transformer-based model to detect antisemitism.

## 🗂️ Project Structure

```
datathon2025_team1/
│
├── data/                           # Raw data files (CSV with tweets)
├── annotations/                   # Annotations CSV following IHRA schema
├── scripts/                       # Scripts for model training
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## 🛠️ Installation

To set up the environment and run the scripts, use:

```bash
pip install -r requirements.txt
```

## 🧠 Challenge #1 – Annotation

- Tweets were collected using Bright Data, filtered with predefined keywords.
- We annotated a set of 100 tweets using the IHRA guidelines.
- Each tweet was labeled using the official IHRA categories.
- Only one annotator completed the process due to team availability.

## ⚙️ Challenge #2 – Modeling (Partial Submission)

- The script in `scripts/model_training.py` shows our planned approach.
- Due to time constraints, we could not fully implement or evaluate the model.
- We propose training a BERTweet model using HuggingFace Transformers.

## ⏳ Known Limitations

- Only 20 tweets were annotated due to team availability.
- No inter-annotator agreement could be measured (one annotator only).
- The model is not trained due to time/resource limitations.

## 👥 Team

Team 1 — Datathon 2025  
Lead Annotator and Contributor: [Seu Nome Aqui]

## 📁 Files To Include

- `data/Team 1 datathon 2025 - bd_20250724_014841_0.csv`
- `annotations/Datathon2025_team1_annotation.csv`
