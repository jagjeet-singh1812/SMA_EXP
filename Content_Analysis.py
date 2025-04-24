from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import sklearn
from sklearn.decomposition import LatentDirichletAllocation

f = open('abstract.txt', 'r')
doc = f.read()


def preprocess(doc):
    lemma = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = doc.lower().split()  # simple tokenization
    final_doc = ""
    for tok in tokens:
        if tok not in stop_words:
            final_doc += lemma.lemmatize(tok) + " "
    final_doc = final_doc.strip()
    print(final_doc)
    return final_doc


# implementing tfid
def keyword_extraction(doc):
    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
    matrix = vectorizer.fit_transform([doc])
    key_words = vectorizer.get_feature_names_out()
    print(f"Top 10 Keywords Extracted from all the docs are : {key_words}")
    scores = matrix.toarray().flatten()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)

    #     plt.barh(key_words, scores, color='skyblue')
    #     plt.xlabel("TF-IDF Score")
    #     plt.title("Top 10 Keywords by TF-IDF")
    #     # plt.gca().invert_yaxis()  # Highest at top
    #     # plt.tight_layout()
    #     plt.show()
    #
    #
    # # def worldcloud_draw():
    #     plt.subplot(1,2,2)
    wor = WordCloud(width=600, height=400, background_color='white').generate(' '.join([doc]))
    #     # plt.figure(figsize=(10, 6))
    #     plt.imshow(wor)
    #     plt.axis('off')
    #     plt.show()
    plt.title("Keyword Extraction")
    plt.subplot(1, 2, 1)
    plt.barh(key_words, scores, color='skyblue')
    plt.xlabel("TF-IDF Score")
    plt.title("Top 10 Keywords by TF-IDF")

    # Right plot: WordCloud
    plt.subplot(1, 2, 2)
    plt.imshow(wor, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud")
    plt.show()


doc = preprocess(doc)
#
# worldcloud_draw()
# keyword_extraction(doc)

count_vectorizer = TfidfVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform([doc])

lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(count_matrix)

# Display Topics
feature_names = count_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
    print(f"Topic {topic_idx + 1}: {top_words}")


