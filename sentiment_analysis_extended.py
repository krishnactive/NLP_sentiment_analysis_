import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from tqdm import tqdm
import nltk
import re
import os
import warnings

warnings.filterwarnings('ignore')
tqdm.pandas()
nltk.download('stopwords')
nltk.download('wordnet')

# 1️⃣ Load datasets
print("📥 Loading datasets...")
trump = pd.read_csv("hashtag_donaldtrump.csv", lineterminator='\n')
biden = pd.read_csv("hashtag_joebiden.csv", lineterminator='\n')

trump['candidate'] = 'trump'
biden['candidate'] = 'biden'
data = pd.concat([trump, biden])
print(f"✅ Combined data shape: {data.shape}")

# 2️⃣ Clean
data.dropna(inplace=True)
data['country'] = data['country'].replace({'United States of America': "US", 'United States': "US"})

# 3️⃣ Sample to speed up
print("🔍 Sampling 20,000 tweets...")
sample = data.sample(20000, random_state=42).copy()

# 4️⃣ Clean tweets
print("🧹 Cleaning tweets...")
stop_words = set(stopwords.words('english'))
lm = WordNetLemmatizer()

def clean(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', str(text))
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    words = text.split()
    words = [lm.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

sample['cleantext'] = sample['tweet'].progress_apply(clean)

# 5️⃣ Sentiment scores
print("📊 Calculating sentiment...")
sample['subjectivity'] = sample['cleantext'].progress_apply(lambda x: TextBlob(x).sentiment.subjectivity)
sample['polarity'] = sample['cleantext'].progress_apply(lambda x: TextBlob(x).sentiment.polarity)
sample['analysis'] = sample['polarity'].apply(lambda x: 'positive' if x>0 else 'negative' if x<0 else 'neutral')

# Filter US tweets
us = sample[sample['country']=='US'].copy()

# 6️⃣ Create plots folder
os.makedirs('plots', exist_ok=True)

# 📊 Plot 1: Sentiment distribution overall
plt.figure(figsize=(6,4))
(us['analysis'].value_counts(normalize=True)*100).plot.bar(color=['green','red','orange'])
plt.ylabel('% of tweets')
plt.title('Sentiment distribution (US tweets)')
plt.tight_layout()
plt.savefig('plots/sentiment_distribution_overall.png')
plt.close()

# 📊 Plot 2: Sentiment by candidate
plt.figure(figsize=(7,5))
sns.countplot(data=us, x='candidate', hue='analysis',
              palette={'positive':'green','negative':'red','neutral':'orange'})
plt.title('Sentiment by Candidate (US tweets)')
plt.ylabel('Number of tweets')
plt.tight_layout()
plt.savefig('plots/sentiment_by_candidate.png')
plt.close()

# 📊 Plot 3: Polarity histogram
plt.figure(figsize=(6,4))
sns.histplot(us['polarity'], bins=50, color='skyblue')
plt.title('Polarity Distribution')
plt.tight_layout()
plt.savefig('plots/polarity_hist.png')
plt.close()

# 📊 Plot 4: Subjectivity histogram
plt.figure(figsize=(6,4))
sns.histplot(us['subjectivity'], bins=50, color='purple')
plt.title('Subjectivity Distribution')
plt.tight_layout()
plt.savefig('plots/subjectivity_hist.png')
plt.close()

# 📊 Plot 5: Most common words (bar)
from collections import Counter
words = ' '.join(us['cleantext']).split()
common_words = Counter(words).most_common(15)
labels, counts = zip(*common_words)
plt.figure(figsize=(7,5))
sns.barplot(y=list(labels), x=list(counts), palette='magma')
plt.title('Top 15 most common words')
plt.xlabel('Frequency')
plt.tight_layout()
plt.savefig('plots/common_words_bar.png')
plt.close()

# 📊 Plot 6: WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(words))
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig('plots/wordcloud.png')
plt.close()

# 📊 Plot 7: Tweets over time
us['date'] = pd.to_datetime(us['created_at'], errors='coerce').dt.date
tweets_per_day = us.groupby('date').size()
plt.figure(figsize=(10,4))
tweets_per_day.plot()
plt.ylabel('Number of tweets')
plt.title('Tweets over time (US)')
plt.tight_layout()
plt.savefig('plots/tweets_over_time.png')
plt.close()

# ✅ Summary
print("\n🎉 All plots saved in 'plots/' folder!")
print("Top sentiment distribution:\n", us['analysis'].value_counts(normalize=True)*100)
print(f"Total US tweets analyzed: {len(us)}")
print(f"Average polarity: {us['polarity'].mean():.3f}")
print(f"Average subjectivity: {us['subjectivity'].mean():.3f}")
