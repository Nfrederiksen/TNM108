#=========================================================
# For course TNM108 at Linköping University
# Mini-project "Clasbot: a FAQ chatbot"
# Written by Nicholas Frederiksen and Pontus Söderqvist
#=========================================================


import pandas as pd
import nltk
import numpy as np
import re
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer # to perform tfidf
from sklearn.metrics import pairwise_distances # to perfrom cosine similarity
from nltk import word_tokenize # to create tokens
from nltk.corpus import stopwords # for stop words
import textwrap

df=pd.read_excel('responsesDB.xlsx')


# function that converts text into lower case and removes special characters

def toLower(x):
    for i in x:
        a=str(i).lower()
        p=re.sub(r'[^a-z0-9åäö]',' ',a)

# function that performs text normalization steps

def text_normalization(text):
    text = str(text).lower()  # text to lower case
    spl_char_text = re.sub(r'[^ a-zåäö]+', '', text)   # removing special characters
    tokens = nltk.word_tokenize(spl_char_text)  # word tokenizing
    stemmer = SnowballStemmer("swedish")
    stem_words = []
    for token in tokens:
        stem_token = stemmer.stem(token)
        stem_words.append(stem_token)

    return " ".join(stem_words)  # returns the lemmatized tokens as a sentence

def remove_stopwords(text):
    stop = stopwords.words('swedish')
    text = text_normalization(text)
    t = []
    new_txt = ""
    splitted_txt = text.split()
    for word in splitted_txt:
        if word in stop:
            continue
        else:
            t.append(word)
        new_txt = " ".join(t)

    return new_txt

def match_answer(array, matrix):
    cos = 1 - pairwise_distances(matrix, array, metric='cosine')  # Applying cosine similarity
    index_value = cos.argmax()
    return index_value, cos.max()

def clasbot():
    # FIXA TILL DATABASEN -------------------------------------------------------------------
    toLower(df['Context'])

    df['stemmed_text_no_stopwords'] = df['Context'].apply(remove_stopwords)
    #df['stemmed_text'] = df['no_stopwords'].apply(text_normalization)  # applying the fuction to the dataset to get clean text

    #df['stemmed_text'] = df['Context'].apply(text_normalization)  # applying the fuction to the dataset to get clean text

    # Using Tf-IDF
    tfidf = TfidfVectorizer()
    tfidf_array = tfidf.fit_transform(df['stemmed_text_no_stopwords']).toarray()  # transforming data into array.
    df_tfidf = pd.DataFrame(tfidf_array, columns=tfidf.get_feature_names())
    # -----------------------------------------------------------------------------------------

    wrapper = textwrap.TextWrapper(width=80)

    print("Hej, jag heter Clasbot! <[^-^]>")
    while True:
        # VAD SÄGER KUNDEN?
        while True:
            try:
                question = input("Vad vill du ha hjälp med? \n")

            except ValueError:
                print("Va?, please try again")
                continue

            if question.lower() == "hej då":
                print("Snälla, gå inte!")
                return

            no_stop = remove_stopwords(question)
            stemmed = text_normalization(no_stop)
            # Using Tf-IDF
            q_tfidf = tfidf.transform([stemmed]).toarray()  # transforming data into array.

            idx, certainty = match_answer(q_tfidf,df_tfidf)

            if certainty > 0.1:
                # Clasbot says this.
                answer = df['Text Response'].loc[idx]
                print("\n", wrapper.fill(answer), "\n --- <[^-^]> \n")
            else:
                print("\n Jag har tyvärr inget bra svar på det. \n Du kan prova att formulera om frågan eller ställ en ny.\n --- <[x_x]> \n")

# Kör Clasbot
clasbot()