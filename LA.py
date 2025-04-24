import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv(('C:/Users/Jagjeet/Downloads/SMA 1-3 Dataset.csv'), encoding='utf-8')

print(df.head(3))

print(df.info())

df = df.dropna(subset=['Location'])

loaction_count = df['Location'].value_counts()
print(loaction_count)

plt.figure(figsize=(12, 6))
plt.suptitle("Location ANALYSIS", fontsize=16, fontweight='bold')
plt.axis('off')

plt.subplot(1, 3, 1)
loaction_count.plot(kind='bar')
# plt.show()
loc = [x for x in df['Location']]
# plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 2)
cloud = WordCloud(width=600, height=300, background_color='white', colormap='Greens').generate(' '.join(loc))
plt.imshow(cloud)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.pie(loaction_count, labels=loaction_count.index, autopct='%1.1f%%')
plt.show()
