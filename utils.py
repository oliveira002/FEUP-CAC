import emoji
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils as ut
import warnings
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

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
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(' '.join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()
