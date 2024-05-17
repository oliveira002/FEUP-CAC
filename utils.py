import emoji
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils as ut
import seaborn as sns
import warnings
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import plotly.express as ex
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

warnings.simplefilter(action='ignore')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

def initial_obs(df):
    display(df.head(10))
    print(f"\n\033[1mAttributes:\033[0m {list(df.columns)}")
    print(f"\033[1mEntries:\033[0m {df.shape[0]}")
    print(f"\033[1mAttribute Count:\033[0m {df.shape[1]}")
    
    print(f"\n\033[1m----Null Count----\033[0m")
    print(df.isna().sum())

def convert_emojis_to_text(text):
    txt = emoji.demojize(text)
    return txt.replace(':', '').replace('_', ' ')


def trim_text(text):
    text = text.strip()
    text = text.split()
    return " ".join(text)
    
def remove_twitter_handles_hashtags(text):
    twitter_handle_pattern = r'@[A-Za-z0-9_]+'

    cleaned_text = re.sub(twitter_handle_pattern, '', text)
    cleaned_text = re.sub(r'#(\w+)', r'\1', cleaned_text)
    
    return cleaned_text

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    stop_words.remove('no')
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

def generate_word_cloud(text, title):
    # Compute TF-IDF values
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(text)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1

    # Map words to TF-IDF scores
    word_scores = {word: score for word, score in zip(feature_names, tfidf_scores)}

    # Generate word cloud with TF-IDF scores
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          prefer_horizontal=0.7).generate_from_frequencies(word_scores)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_sentiments(df):
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()
    
def common_words(df, num):
    all_texts = [' '.join(text) for text in df['lemmatized_text']]
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1

    word_scores = {word: score for word, score in zip(feature_names, tfidf_scores)}

    top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:num]
    top_words_df = pd.DataFrame(top_words, columns=['Common Words', 'TF-IDF Score'])
    
    fig = ex.treemap(top_words_df, path=['Common Words'], values='TF-IDF Score',
                     title=f"{num} Most Common Words In Tweets (TF-IDF Weighted)")
    fig.show()

def pre_process_pipeline(df, attribute):
    df[attribute] = df[attribute].apply(trim_text)
    df[attribute] = df[attribute].apply(contractions.fix)
    df[attribute] = df[attribute].apply(lambda x:re.sub(r"http\S+", "", x))
    df[attribute] = df[attribute].apply(convert_emojis_to_text)
    df[attribute] = df[attribute].apply(remove_twitter_handles_hashtags)
    df[attribute] = df[attribute].apply(remove_special_characters)
    df[attribute] = df[attribute].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))
    df[attribute] = df[attribute].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    df[attribute] = df[attribute].str.lower()
    df['tokenized_text'] = df[attribute].apply(lambda x: word_tokenize(x))
    df['tokenized_text'] = df['tokenized_text'].apply(remove_stopwords)
    
    lemmatizer = WordNetLemmatizer()
    df['lemmatized_text'] = df['tokenized_text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    df['clean_text'] = df['lemmatized_text'].apply(lambda text: " ".join(text))
    
    return df
    
    
def generate_features(train_data,test_data, feature_type='tfidf', **kwargs):
    train_text = train_data['clean_text'].astype(str).values
    test_text = test_data['clean_text'].astype(str).values
    
    if feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(**kwargs)
    elif feature_type == 'bow':
        vectorizer = CountVectorizer(**kwargs)
    else:
        raise ValueError("Invalid feature_type. Choose from 'tfidf' or 'bow'.")
    
    train_features = vectorizer.fit_transform(train_text)
    test_features = vectorizer.transform(test_text)
    
    return train_features, test_features

def predict_labels(train_features, train_labels, test_features):
    nb_model = MultinomialNB()
    nb_model.fit(train_features, train_labels)
    
    pred_nb = nb_model.predict(test_features)
    
    svd = TruncatedSVD(n_components=100)
    train_features_svd = svd.fit_transform(train_features)
    
    svm_model = SVC()
    svm_model.fit(train_features_svd, train_labels)
    
    test_features_svd = svd.transform(test_features)
    
    pred_svm = svm_model.predict(test_features_svd)
    
    return pred_nb, pred_svm

def get_top_ngrams(corpus, n, ngram_size):
    vectorizer = CountVectorizer(ngram_range=(ngram_size, ngram_size)).fit(corpus)
    word_matrix = vectorizer.transform(corpus)
    word_sum = word_matrix.sum(axis=0)
    word_freq = [(word, word_sum[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    sorted_word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)
    return sorted_word_freq[:n]

def plot_n_grams(df_tweets, n_grams):
    fig, ax = plt.subplots(1, 3, figsize=(10,5))
    
    sentiments = df_tweets['sentiment'].unique()
    colors = ['blue', 'red', 'green']
    
    for i, sentiment in enumerate(sentiments):
        sent_df = df_tweets[df_tweets['sentiment'] == sentiment]
        most_common_bi = ut.get_top_ngrams(sent_df['token_text'],10,n_grams)
        most_common_bi = dict(most_common_bi)
        sns.barplot(x=list(most_common_bi.values()),y=list(most_common_bi.keys()), ax=ax[i])
        ax[i].set_title(sentiment, fontsize=20, color=colors[i])

    plt.suptitle(f'{str(n_grams)}-Gram of each sentiment', fontsize=25)
    plt.tight_layout()
    plt.show()

def plot_avg_word_length_distribution_multi(*dfs):
    fig, axs = plt.subplots(1, len(dfs), figsize=(12, 8), sharey=True, sharex=True)
    sentiments = ["Positive", "Neutral", "Negative"]
    colors = ['green', 'blue', 'red']
    fig.suptitle('Average Word Length in Each Text', fontsize=25)

    for i, df in enumerate(dfs):
        # Tokenize the text and calculate the average word length
        avg_word_length = df['text'].apply(lambda x: [len(word) for word in x.split()])
        avg_word_length = avg_word_length.apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)

        # Plot the distribution
        sns.histplot(avg_word_length, ax=axs[i], color=colors[i])
        axs[i].set_title(sentiments[i], fontsize=20)
        axs[i].set_xlabel('Average Word Length')
        axs[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.xlim([0, 20])  # Adjust the x-axis limits if needed
    plt.show()

def get_tweets_by_date(df, date):
    """
    Function to retrieve tweets for a specific date.
    
    Parameters:
    - df: DataFrame containing the tweets.
    - date: String in 'YYYY-MM-DD' format or a datetime.date object.
    
    Returns:
    - DataFrame containing tweets for the specified date.
    """
    # Convert input date to datetime.date if it's a string
    if isinstance(date, str):
        date = pd.to_datetime(date).date()
    
    # Filter the DataFrame for the specified date
    tweets_on_date = df[df['date_'] == date]
    
    return tweets_on_date[['orig_text','clean_text']]

def topic_modelling(df):
    tfidf_vectorizer = TfidfVectorizer(max_features = 40000, max_df=0.95, min_df=2)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])

    
    num_topics = 4 
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(tfidf_matrix)


    def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print(f"Topic {topic_idx+1}:")
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))

    no_top_words = 15  # Number of top words to display for each topic
    feature_names = tfidf_vectorizer.get_feature_names_out()
    display_topics(lda, feature_names, no_top_words)