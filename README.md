# 🐦 NLP Sentiment Analysis for US Election

Analyze and predict the sentiment of tweets related to the **2020 US Presidential Election** using Natural Language Processing.

> ⚡ Powered by Python, pandas, NLTK, TextBlob, Plotly, and Matplotlib.

---

## 📌 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Setup & Installation](#-setup--installation)



---

## ✨ Features
- Data preprocessing & cleaning
- Exploratory data analysis (EDA)
- Sentiment analysis (positive, negative, neutral)
- Word clouds
- Country-wise tweet analysis
- Visualizations using Plotly

---

## 📂 Project Structure
NLP_sentiment_analysis/
├── data/
│ ├── hashtag_donaldtrump.csv # (not included, see Dataset)
│ ├── hashtag_joebiden.csv # (not included, see Dataset)
├── sentiment_analysis.py
├── requirements.txt
├── README.md
└── .gitignore


---

## 📥 Dataset
  
Download them from:
- 📦 **[US Election 2020 Tweets on Kaggle](https://www.kaggle.com/datasets/subho117/nlp-sentiment-analysis-for-us-election?select=hashtag_donaldtrump.csv)**  

After downloading, place the files in the `data/` folder:
- `hashtag_donaldtrump.csv`
- `hashtag_joebiden.csv`

---

## ⚙️ Setup & Installation

```bash
# Clone this repo
git clone https://github.com/krishnactive/NLP_sentiment_analysis_.git
cd NLP_sentiment_analysis

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
python sentiment_analysis.py

