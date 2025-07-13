# sentiment_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
import os
import warnings

warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('wordnet')

# 0️⃣ Create plots folder
os.makedirs('plots', exist_ok=True)

# 1️⃣ Load datasets
trump = pd.read_csv("hashtag_donaldtrump.csv", lineterminator='\n')
biden = pd.read_csv("hashtag_joebiden.csv", lineterminator='\n')

# 2️⃣ Add candidate column and combine
trump['candidate'] = 'trump'
biden['candidate'] = 'biden'
data = pd.concat([trump, biden])
print(f"✅ Combined data shape: {data.shape}")

# 3️⃣ Clean data
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

# 6️⃣ Filter US tweets and apply analysis
us_data = data[data['country'] == 'US'].copy()
us_data['polarity'] = us_data['cleantext'].apply(get_polarity)
us_data['subjectivity'] = us_data['cleantext'].apply(get_subjectivity)
us_data['analysis'] = us_data['polarity'].apply(get_analysis)

print("\n📊 Sentiment distribution:")
print(us_data['analysis'].value_counts(normalize=True)*100)

# 7️⃣ Plot: Distribution of sentiments
plt.style.use('dark_background')
colors = ['orange', 'green', 'red']
plt.figure(figsize=(7,5))
(us_data['analysis'].value_counts(normalize=True) * 100).plot.bar(color=colors)
plt.ylabel("Percentage of tweets")
plt.title("Distribution of Sentiments (US Tweets)")
plt.tight_layout()
plt.savefig('plots/sentiment_distribution.png')
plt.close()

# 8️⃣ Plot: Polarity histogram
plt.figure(figsize=(7,5))
sns.histplot(us_data['polarity'], bins=30, color='purple', kde=True)
plt.title('Polarity Distribution')
plt.xlabel('Polarity')
plt.tight_layout()
plt.savefig('plots/polarity_distribution.png')
plt.close()

# 9️⃣ Plot: Subjectivity histogram
plt.figure(figsize=(7,5))
sns.histplot(us_data['subjectivity'], bins=30, color='teal', kde=True)
plt.title('Subjectivity Distribution')
plt.xlabel('Subjectivity')
plt.tight_layout()
plt.savefig('plots/subjectivity_distribution.png')
plt.close()

# 🔟 Plot: Average polarity per candidate
avg_polarity = us_data.groupby('candidate')['polarity'].mean()
plt.figure(figsize=(6,5))
avg_polarity.plot(kind='bar', color=['blue', 'red'])
plt.ylabel('Average Polarity')
plt.title('Average Polarity per Candidate (US Tweets)')
plt.tight_layout()
plt.savefig('plots/avg_polarity_per_candidate.png')
plt.close()

# 1️⃣1️⃣ Plot: Number of tweets per candidate
plt.figure(figsize=(6,5))
us_data['candidate'].value_counts().plot(kind='bar', color=['red', 'blue'])
plt.ylabel('Number of Tweets')
plt.title('Number of US Tweets per Candidate')
plt.tight_layout()
plt.savefig('plots/tweets_per_candidate.png')
plt.close()

print("✅ All plots saved in 'plots/' folder!")
