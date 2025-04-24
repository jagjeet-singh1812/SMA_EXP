import pandas as pd
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv('SMA 1-3 Dataset.csv')
print(df.head())


def senti(text):
    st = str(text)
    t = TextBlob(st)
    return t.sentiment.polarity


def score(x):
    if x > 0:
        return "pos"
    elif x < 0:
        return "neg"
    else:
        return "neutral"


df['senti'] = df['Content'].apply(senti)
print(df.head(10))
df['pol'] = df['senti'].apply(score)
print(df['pol'])
dfx = df['pol'].value_counts()

plt.figure(figsize=(10, 6))
# dfx.plot(kind='bar', color=['red', 'green', 'yellow'])
dfx.plot(kind='pie', labels=['pos', 'neg', 'neu'])

# sns.countplot(data=df, x='pol', hue=None, palette='coolwarm', legend=False)
# plt.suptitle('All plot', fontsize=20, fontweight='bold')
plt.show()


pos = ''.join(df[df['pol'] == 'pos']['Content'])
print(pos)

w = WordCloud(width=400, height=200, background_color='white', colormap='Greens').generate(pos)

plt.figure(figsize=(10, 6))
plt.imshow(w)
plt.show()

neg = ''.join(df[df['pol'] == 'neg']['Content'])
w = WordCloud(width=400, height=200, background_color='black', colormap='Reds').generate(neg)

plt.figure(figsize=(10, 6))
plt.imshow(w)
plt.show()
