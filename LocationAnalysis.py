import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

df = pd.read_csv('SMA 1-3 Dataset.csv')
print(df.head(10))
print(df.info())
print(df.describe())

print("Null values in Location:", df['Location'].isnull().sum())
df = df.dropna(subset=['Location'])

values = df['Location'].value_counts()
# values = values.head(3)
# values.sort_values()
# print(values)
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.title("top 7 locations")
values.plot(kind='bar')
plt.xticks(rotation=45)

loc = [i for i in df['Location']]
# location=[x for x in df["Location"]]

cloud = WordCloud(width=400, height=200, background_color='white').generate(" ".join(loc))

plt.subplot(2, 2, 2)
plt.imshow(cloud)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cloud)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.pie(values, labels=values.index, autopct='%1.1f%%')
plt.show()
