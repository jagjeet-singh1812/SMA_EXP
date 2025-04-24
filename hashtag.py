import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv('SMA 1-3 Dataset.csv')
print(df['Hashtags'])

df['hash'] = df['Hashtags'].apply(lambda x: x.lower().split())
print(df['hash'])

list_of_has = []
for i in df['hash']:
    for tags in i:
        list_of_has.append(tags)

print(list_of_has)

counts = Counter(list_of_has)

print(counts)
whole = counts.most_common(5)

data = pd.DataFrame(whole, columns=['hash', 'count'])
print(data)
# d2 = pd.Series(whole, columns=['hash', 'count'])
# plt.figure(figsize=(10, 5))
# c = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(counts)
# plt.imshow(c)
# plt.show()
#
plt.figure(figsize=(10, 5))
# plt.bar(data['hash'], data['count'])
# plt.show()

plt.pie(data['count'], labels=data['hash'], autopct='%1.1f%%')
plt.show()
