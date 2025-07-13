# sentiment_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
import warnings

warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('wordnet')

# 1️⃣ Load datasets
trump = pd.read_csv("hashtag_donaldtrump.csv", lineterminator='\n')
biden = pd.read_csv("hashtag_joebiden.csv", lineterminator='\n')

# 2️⃣ Add candidate column and combine
trump['candidate'] = 'trump'
biden['candidate'] = 'biden'
data = pd.concat([trump, biden])
print(f"Combined data shape: {data.shape}")

# 3️⃣ Basic cleaning
data.dropna(inplace=True)
data['country'] = data['country'].replace({'United States of America': "US", 'United States': "US"})

# 4️⃣ Clean tweets
def clean(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', str(text))
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    words = text.split()
    lm = WordNetLemmatizer()
    words = [lm.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    return ' '.join(words)

data['cleantext'] = data['tweet'].apply(clean)

# 5️⃣ Sentiment functions
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def get_analysis(score):
    if score < 0:
        return 'negative'
    elif score == 0:
        return 'neutral'
    else:
        return 'positive'

# 6️⃣ Apply sentiment analysis on US tweets
us_data = data[data['country'] == 'US'].copy()
us_data['subjectivity'] = us_data['cleantext'].apply(get_subjectivity)
us_data['polarity'] = us_data['cleantext'].apply(get_polarity)
us_data['analysis'] = us_data['polarity'].apply(get_analysis)

# 7️⃣ Plot distribution of sentiments
plt.style.use('dark_background')
colors = ['orange', 'green', 'red']
plt.figure(figsize=(7,5))
(us_data['analysis'].value_counts(normalize=True) * 100).plot.bar(color=colors)
plt.ylabel("Percentage of tweets")
plt.title("Distribution of Sentiments (US Tweets)")
plt.show()

print("\nSentiment distribution:")
print(us_data['analysis'].value_counts(normalize=True)*100)
