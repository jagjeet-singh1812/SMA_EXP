import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('SMA 1-3 Dataset.csv')
print(df.info())

print(df.head(4))

df['Date'] = pd.to_datetime(df['Post_Date'], errors='coerce')

print(df.head())
# see this is imp
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Day'] = df['Date'].dt.day
df['Month_name'] = df['Date'].dt.month_name()
df['Day_name'] = df['Date'].dt.day_name()
print(df.head())

df2 = pd.read_csv('youtube_womens_safety_full_data(in).csv')
print(df2['published_at'])

# df2 = df2.dropna(subset=['published_at'])

df2['Date'] = pd.to_datetime(df2['published_at'], errors='coerce')  # imp for handling NAT
df2 = df2.dropna(subset=['Date'])
# print(df2['published_at'].isnull().sum())

print(df2['published_at'])
# print(df2.describe())

# analysis starts from Here>>>>
#
# monthly_engagement = df.groupby(['Month_name', 'Month', 'Year'])['Likes'].sum().reset_index()
# monthly_engagement = monthly_engagement.sort_values(by=['Year', 'Month'])
# monthly_engagement = monthly_engagement.drop(columns='Month')
# print(monthly_engagement)

# plt.figure(figsize=(18, 16))
# # monthly_engagement.plot(kind='line', marker='o', color='green')
# plt.plot(monthly_engagement['Month_name'] + ' ' + monthly_engagement['Year'].astype(str),
#          monthly_engagement['Likes'], marker='o', color='green')
# plt.show()
# plt.subplot(1, 4, 1)
monthly_eng = df.groupby(['Month', 'Year'])['Likes'].mean()

# print(monthly_eng['Likes'])
# monthly_eng.plot(kind='line', marker='o')
# plt.xticks(rotation=90)
# plt.subplot(1, 4, 2)
# monthly_eng.plot(kind='area')
# plt.xticks(rotation=90)
#
# plt.subplot(1, 4, 3)
# monthly_eng.plot(kind='bar')
# plt.xticks(rotation=90)
# plt.show()
#
# total_likes = sum(monthly_eng['Like'])
# likes_by_month = monthly_eng['Like']
# monthly_eng_major = total = likes_by_month.sum()
# likes_percent = (likes_by_month / total) * 100
#
# # Filter major contributors
# major = likes_by_month[likes_percent > 5]
# minor = likes_by_month[likes_percent <= 5]
#
# plt.subplot(1, 4, 4)
# monthly_eng.plot(kind='pie')
# plt.xticks(rotation=90)
#
# plt.show()
#
#
# day_wise_eng = df.groupby(['Day_name'])['Likes'].sum()
# print(day_wise_eng)
# plt.figure(figsize=(10,6))
# day_wise_eng.plot(kind='bar')
# plt.show()


vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
matrix_score = vectorizer.fit_transform(df['Content'])
names = vectorizer.get_feature_names_out()
# print(matrix_score)
# print(names)

score = matrix_score.sum(axis=0).A1
# score2=matrix_score.toarray().flatten()
# print(score)
plt.figure(figsize=(10, 6))
plt.bar(names, score)
plt.show()
