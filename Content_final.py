import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import regex as re
import gensim
from gensim.corpora import Dictionary


#
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')


def preprocess(doc):
    stop_words = set(stopwords.words('english'))
    # token = word_tokenize(doc)
    doc = re.sub(r'http\S+|www\S+|https\S+', '', doc)

    # Remove @mentions and hashtags
    doc = re.sub(r'@\w+|#\w+', '', doc)

    # Remove retweet symbols (RT)
    doc = re.sub(r'\brt\b', '', doc)

    # Remove special characters, digits, punctuation
    doc = re.sub(r'[^a-z\s]', '', doc)
    token = word_tokenize(doc.lower())
    # print(f"Tokenized Words Are: {token}")

    lemma = WordNetLemmatizer()
    cleaned_doc = ""
    for t in token:
        if t not in stop_words:
            cleaned_doc += lemma.lemmatize(t) + " "
    cleaned_doc.strip()

    return cleaned_doc


#
# def Content_Analysis(doc):
#     vectorize = TfidfVectorizer(max_features=10)
#     matrix = vectorize.fit_transform([doc])
#
#     keywords = vectorize.get_feature_names_out()
#     print(f'The Keywords extracted are {keywords}')
#
#     #  for keyword extraction
#
#     cloud = WordCloud(width=800, height=600, background_color='white').generate(" ".join([doc]))
#     plt.figure(figsize=(10, 6))
#
#     plt.imshow(cloud)
#     plt.axis('off')
#     plt.show()
#
#     #     topic modelling
#     count_vectorizer = CountVectorizer()
#     count_matrix = count_vectorizer.fit_transform([doc])
#
#     lda = LatentDirichletAllocation(n_components=3, random_state=42)
#     lda.fit(count_matrix)
#
#     # Display Topics
#     feature_names = count_vectorizer.get_feature_names_out()
#     for topic_idx, topic in enumerate(lda.components_):
#         top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
#         print(f"Topic {topic_idx + 1}: {top_words}")
#
#     plt.figure(figsize=(10, 6))
#     plt.barh()
#
def using_gen(doc):
    w = doc.lower().split()
    print("Tokenized:", w)

    documents = [w]  # Gensim expects a list of documents (each as a list of tokens)
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(w)]  # Only one document here

    lda = gensim.models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)
    topics = lda.print_topics()

    for i, t in topics:
        print(f"Topic #{i}: {t}")


def Content_Analysis(doc):
    # TF-IDF Keyword Extraction
    vectorizer = TfidfVectorizer(max_features=10)
    matrix = vectorizer.fit_transform([doc])
    keywords = vectorizer.get_feature_names_out()
    score = matrix.toarray().flatten()
    scores = matrix.sum(axis=0).A1

    print(f'Top 10 Keywords extracted are: {keywords}')

    # Plot TF-IDF keywords
    plt.figure(figsize=(10, 5))
    plt.barh(keywords, scores, color='skyblue')
    plt.xlabel("TF-IDF Score")
    plt.title("Top 10 Keywords by TF-IDF")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Word Cloud
    cloud = WordCloud(width=800, height=600, background_color='white').generate(doc)
    plt.figure(figsize=(10, 6))
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of Cleaned Text")
    plt.tight_layout()
    plt.show()

    # Topic Modelling with LDA
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform([doc])
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(count_matrix)

    feature_names = count_vectorizer.get_feature_names_out()
    plt.figure(figsize=(8, 4))
    i = 0
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-3:-1]
        top_words = [feature_names[i] for i in top_indices]
        top_scores = topic[top_indices]
        print(f"Topic {topic_idx + 1}: {top_words}")
        plt.subplot(1, 3, i + 1)
        plt.barh(top_words, top_scores, color='salmon')
        plt.xlabel("Word Importance")
        plt.title(f"Topic {topic_idx + 1} Keywords")

        i += 1
    plt.show()


url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
df = pd.read_csv(url)
df = df.head(500)
# print(df.head(10))
# print(df.info())
# print(df.describe())
# df.info()
# df.describe()


df['cleaned_content'] = df['tweet'].astype(str).apply(preprocess)

print(df['cleaned_content'])
print(df.head(10))

all_text = ' '.join(df['cleaned_content'])
Content_Analysis(all_text)
# using_gen(all_text)
