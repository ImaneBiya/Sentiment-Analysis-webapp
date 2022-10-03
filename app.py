from flask import Flask, render_template, request, url_for, redirect, session
import pymongo
import bcrypt
from flask import Flask, render_template, request, redirect, url_for
import os
from os.path import join, dirname, realpath
import pymongo
import pandas as pd
import json
import os
from werkzeug.utils import secure_filename
from flask_session import Session


# for text preprocessing
import re
import nltk
import spacy
nlp = spacy.load('en_core_web_sm')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

# import vectorizers
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# import numpy for matrix operation
import numpy as np

# import LDA from sklearn
from sklearn.decomposition import LatentDirichletAllocation
import io
import tweepy
import pymongo
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import string
import matplotlib.pyplot as plt
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from pymongo import MongoClient, ASCENDING, TEXT
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import re
import sys
import nltk
from nltk.corpus import stopwords, sentiwordnet as swn
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import collections

import string
import matplotlib.pyplot as plt
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = "testing"
client = pymongo.MongoClient("mongodb+srv://imane:ImaneB1999@newcluster.zq40hze.mongodb.net/test")
db = client.get_database('total_records')
records = db.register
SESSION_TYPE = 'redis'
app.config.from_object(__name__)
sess = Session()
sess.init_app(app)


# Sentiment analysis function using VADER
def vader_sentiment_scores1(data_frame):
    # Define SentimentIntensityAnalyzer object of VADER.
    SID_obj = SentimentIntensityAnalyzer()

    # calculate polarity scores which gives a sentiment dictionary,
    # Contains pos, neg, neu, and compound scores.
    sentiment_list = []
    for row_num in range(len(data_frame)):
        sentence = data_frame['Tweets'][row_num]

        polarity_dict = SID_obj.polarity_scores(sentence)

        # Calculate overall sentiment by compound score
        if polarity_dict['compound'] >= 0.05:
            sentiment_list.append("Positive 😊 😊")


        elif polarity_dict['compound'] <= - 0.05:
            sentiment_list.append("Negative 🙁 🙁")

        else:
            sentiment_list.append("Neutral  😐 😐")

    data_frame['Sentiment'] = sentiment_list

    return data_frame
def vader_sentiment_scores2(sentence):
    # Define SentimentIntensityAnalyzer object of VADER.
    SID_obj = SentimentIntensityAnalyzer()

    polarity_dict = (SID_obj.polarity_scores(str(sentence)))

    # Calculate overall sentiment by compound score
    if polarity_dict['compound'] >= 0.05:
        label = "Positive 😊 😊"
    elif polarity_dict['compound'] <= - 0.05:
        label = "Negative 🙁 🙁"
    else:
        label = "Neutral  😐 😐"


# Sentiment analysis function using VADER
def vader_sentiment_scores(data_frame):
    # Define SentimentIntensityAnalyzer object of VADER.
    SID_obj = SentimentIntensityAnalyzer()

    # calculate polarity scores which gives a sentiment dictionary,
    # Contains pos, neg, neu, and compound scores.
    sentiment_list = []
    for row_num in range(len(data_frame)):
        sentence = data_frame['Reviews'][row_num]

        polarity_dict = (SID_obj.polarity_scores(str(sentence)))

        # Calculate overall sentiment by compound score
        if polarity_dict['compound'] >= 0.05:
            sentiment_list.append("Positive 😊 😊")

        elif polarity_dict['compound'] <= - 0.05:
            sentiment_list.append("Negative 🙁 🙁")

        else:
            sentiment_list.append("Neutral 😐 😐")

    data_frame['Sentiment'] = sentiment_list

    return data_frame





@app.route("/", methods=['post', 'get'])
def index():
    message = ''

    if request.method == "POST":
        user = request.form.get("fullname")
        email = request.form.get("email")

        password1 = request.form.get("password1")
        password2 = request.form.get("password2")

        user_found = records.find_one({"name": user})
        email_found = records.find_one({"email": email})
        if user_found:
            message = 'There already is a user by that name'
            return render_template('index.html', message=message)
        if email_found:
            message = 'This email already exists in database'
            return render_template('index.html', message=message)
        if password1 != password2:
            message = 'Passwords should match!'
            return render_template('index.html', message=message)
        else:
            hashed = bcrypt.hashpw(str(password2).encode('utf-8'), bcrypt.gensalt())
            user_input = {'name': user, 'email': email, 'password': hashed}
            records.insert_one(user_input)

            user_data = records.find_one({"email": email})
            new_email = user_data['email']

            return render_template('logged_in.html', email=new_email)
    return render_template('index.html')


@app.route('/logged_in')
def logged_in():
    if "email" in session:
        email = session["email"]
        return render_template('logged_in.html', email=email)
    else:
        return redirect(url_for("login"))

@app.route("/login", methods=["POST", "GET"])
def login():
    message = 'Please login to your account'


    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        email_found = records.find_one({"email": email})
        if email_found:
            email_val = email_found['email']
            passwordcheck = email_found['password']

            if bcrypt.checkpw(password.encode('utf-8'), passwordcheck):
                session["email"] = email_val
                return redirect(url_for('logged_in'))
            else:
                if "email" in session:
                    return redirect(url_for("logged_in"))
                message = 'Wrong password'
                return render_template('login.html', message=message)
        else:
            message = 'Email not found'
            return render_template('login.html', message=message)
    return render_template('login.html', message=message)


@app.route("/logout", methods=["POST", "GET"])
def logout():
    if "email" in session:
        session.pop("email", None)
        return render_template("signout.html")
    else:
        return render_template('index.html')


# Define folder to save uploaded files to process further
UPLOAD_FOLDER = os.path.join('static', 'files')

# Define allowed files (for this example I want only csv file)
ALLOWED_EXTENSIONS = {'csv'}


# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/upload')
def upload():
    return render_template('index_upload_data.html')


@app.route('/upload', methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # upload file flask
        uploaded_df = request.files['uploaded-file']

        # Extracting uploaded data file name
        data_filename = secure_filename(uploaded_df.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
        # flask upload file to database (defined uploaded folder in static path)
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))

        # Storing uploaded file path in flask session
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
        parseCSV(file_path)
        return render_template('index_upload_data_page2.html')


def parseCSV(filePath):
    # Use Pandas to parse the CSV file
    csvData = pd.read_csv(filePath)
    data = csvData.to_dict(orient="records")
    db = client["Data"]

    db.db[filePath].insert_many(data)

@app.route('/show_data')
def showData():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)

    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    uploaded_df = pd.read_csv(data_file_path)

    # pandas dataframe to html table flask
    uploaded_df_html = uploaded_df.to_html()
    return render_template('show_csv_data.html', data_var=uploaded_df_html)

# ENV vars
consumerKey = '2JyRfBrlR4P0p7dHtlxJC5n3i'
consumerSecret = '2EYghUtMAGb691qpQF9Bbu5LVQViYvQvjI1Nz7f74EJ6QcKwo2'
accessToken = '1518234480422629376-IGQXIzr9cryTkbcEC38SgR1CIgNryK'
accessTokenSecret = 'cqJIcORSzQw7vhwyZ5NY2LbKHsQ3KaVWoITxe8YhVNZ1t'

# Create the authentification object
authenticate=tweepy.OAuthHandler(consumerKey, consumerSecret)
# Set the access token and the access token secret
authenticate.set_access_token(accessToken, accessTokenSecret)
# Create the API object
api= tweepy.API(authenticate, wait_on_rate_limit=True)

client = pymongo.MongoClient("mongodb+srv://imane:ImaneB1999@newcluster.zq40hze.mongodb.net/test")

@app.route('/scrap', methods=["GET","POST"])
def tweets():
    return render_template('scrap.html')

# Get tweets from twitter


@app.route('/tweets', methods=["GET","POST"])
def get_tweets():
   if request.method == "GET":
    inp = request.form.get("inp")
    search_term = [inp]
    # Create a cursor object
    tweets = api.search(q=search_term, lang='en', tweet_mode='extended', count=5000)
    # Store the tweets in a variable and get the full text
    all_tweets = [tweet.full_text for tweet in tweets]
    # Create a dataframe to store the tweets with a column called 'tweets'
    df = pd.DataFrame(all_tweets, columns=['Tweets'])
    tweets = df.to_dict(orient="records")
    db = client["tweets"]

    db.db[search_term].insert_many(tweets)

    uploaded_df_sentiment = vader_sentiment_scores1(df)
    pos = 0
    neg = 0
    neut = 0
    for row_num in range(len(uploaded_df_sentiment)):
           if uploaded_df_sentiment['Sentiment'][row_num] == "Positive 😊 😊":
               pos = pos + 1
           elif uploaded_df_sentiment['Sentiment'][row_num] == "Negative 🙁 🙁":
               neg = neg+1
           else :
               neut = neut+1

    uploaded_df_html = uploaded_df_sentiment.to_html()
    return render_template('tweets.html', data=uploaded_df_html, pos=pos, neg=neg, neut=neut)

@app.route('/read')
def read():
    return render_template('index_upload_and_polarity_data.html')


@app.route('/read', methods=("POST", "GET"))
def readFile():
    if request.method == 'POST':
        uploaded_file = request.files['uploaded-file']
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        session['uploaded_csv_file'] = df.to_json()
        return render_template('index_upload_and_polarity_data_page2.html')

@app.route('/show')
def show():
    # Get uploaded csv file from session as a json value
    uploaded_json = session.get('uploaded_csv_file', None)
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(eval(uploaded_json))

    # Convert dataframe to html format
    uploaded_df_html = uploaded_df.to_html()
    return render_template('show.html', data=uploaded_df_html)


@app.route('/sentiment')
def Sentiment():
    # Get uploaded csv file from session as a json value
    uploaded_json = session.get('uploaded_csv_file', None)
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(eval(uploaded_json))
    # Apply sentiment function to get sentiment score
    uploaded_df_sentiment = vader_sentiment_scores(uploaded_df)
    pos = 0
    neg = 0
    neut = 0
    for row_num in range(len(uploaded_df_sentiment)):
        if uploaded_df_sentiment['Sentiment'][row_num] == "Positive 😊 😊":
            pos = pos + 1
        elif uploaded_df_sentiment['Sentiment'][row_num] == "Negative 🙁 🙁":
            neg = neg + 1
        else:
            neut = neut + 1
    uploaded_df_html = uploaded_df_sentiment.to_html()
    return render_template('sentiment.html', data=uploaded_df_html, pos=pos, neg=neg, neut=neut)

@app.route('/explore')
def preprocess():
    # Get uploaded csv file from session as a json value
    uploaded_json = session.get('uploaded_csv_file', None)
    # Convert json to data frame
    df = pd.DataFrame.from_dict(eval(uploaded_json))




    def remove_punctuations(text):
        for punctuation in string.punctuation:
            text = text.replace(punctuation, '')
        return text
    df['no_punc'] = df['Reviews'].apply(remove_punctuations)

    df['tokenized'] = df['no_punc'].apply(word_tokenize)

    df['lower'] = df['tokenized'].apply(lambda x: [word.lower() for word in x])

    stop_words = set(stopwords.words('english'))

    df['stopwords_removed'] = df['lower'].apply(lambda x: [word for word in x if word not in stop_words])

    df['pos_tags'] = df['stopwords_removed'].apply(nltk.tag.pos_tag)


    LR = r"""
          NP: {<NN><JJ>}
              {<JJ><NN>}
              {<VBP><NN>}
              {<JJ><NN><NN>}
              {<RB><JJ><NN>}
              {<NN><NN><JJ>}
              {<JJ><JJ><NN>}
              {<RB><RB><NN>}
              {<JJ><NN><JJ>}
              {<VBP><JJ><NN>}
              {<NN><RB><JJ>}
              {<NN><JJ><NN>}
              {<VBP><NNS>}
              {<NNS><JJ>}
              {<NNS><NN>}
              {<NNP><JJ>}
              {<NN><NNP>}
              {<NNP><NN>}
        """
    nv = []
    cp = nltk.RegexpParser(LR)
    for i in df.index:
        tags = df['pos_tags'][i]
        result = cp.parse(tags)
        for j in result.subtrees():
                if j.label() == 'NP':
                    ner = nltk.ne_chunk(j)
                    nv.append(ner)
    data = pd.DataFrame(nv)
    stt = []
    for i in range(len(data)):
        stt.append(data[0][i][0])

    st = []
    for i in range(len(data)):
        st.append(data[1][i][0])

    a = []
    for i in range(len(stt)):
        a.append(f'{stt[i]} {st[i]}')

    d = pd.DataFrame(a, columns=['Reviews'])

    uploaded_df_sentiment = vader_sentiment_scores(d)
    pos = 0
    neg = 0
    neut = 0
    for row_num in range(len(uploaded_df_sentiment)):
        if uploaded_df_sentiment['Sentiment'][row_num] == "Positive 😊 😊":
            pos = pos + 1
        elif uploaded_df_sentiment['Sentiment'][row_num] == "Negative 🙁 🙁":
            neg = neg + 1
        else:
            neut = neut + 1
    uploaded_df_html = uploaded_df_sentiment.to_html()

    return render_template('explore.html', data=uploaded_df_html, pos=pos, neg=neg, neut=neut)

@app.route('/LDA', methods=['GET', 'POST'])
def dataupload():
        # Get uploaded csv file from session as a json value
        uploaded_json = session.get('uploaded_csv_file', None)
        # Convert json to data frame
        df = pd.DataFrame.from_dict(eval(uploaded_json))

        def remove_punctuations(text):
            for punctuation in string.punctuation:
                text = text.replace(punctuation, '')
            return text

        df['no_punc'] = df['Reviews'].apply(remove_punctuations)
        stop_words = set(stopwords.words('english'))
        df['tokenized'] = df['no_punc'].apply(word_tokenize)
        df['lower'] = df['tokenized'].apply(lambda x: [word.lower() for word in x])
        df['stopwords_removed'] = df['lower'].apply(lambda x: [word for word in x if word not in stop_words])

        df['pos_tags'] = df['stopwords_removed'].apply(nltk.tag.pos_tag)

        def get_wordnet_pos(tag):
            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            elif tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN

        df['wordnet_pos'] = df['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])

        wnl = WordNetLemmatizer()

        df['lemmatized'] = df['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])

        # Converting text into numerical representation
        tf_idf_vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)

        # Converting text into numerical representation
        cv_vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)

        # Array from TF-IDF Vectorizer
        tf_idf_arr = tf_idf_vectorizer.fit_transform(df['lemmatized'])

        # Array from Count Vectorizer
        cv_arr = cv_vectorizer.fit_transform(df['lemmatized'])

        # Creating vocabulary array which will represent all the corpus
        vocab_tf_idf = tf_idf_vectorizer.get_feature_names()

        # Creating vocabulary array which will represent all the corpus
        vocab_cv = cv_vectorizer.get_feature_names()

        # Implementation of LDA:

        # Create object for the LDA class
        # Inside this class LDA: define the components:
        lda_model = LatentDirichletAllocation(n_components=20, max_iter=20, random_state=20)

        # fit transform on model on our count_vectorizer : running this will return our topics
        X_topics = lda_model.fit_transform(tf_idf_arr)

        # .components_ gives us our topic distribution
        topic_words = lda_model.components_

        #  Define the number of Words that we want to print in every topic : n_top_words
        n_top_words = 15
        nv = []
        for i, topic_dist in enumerate(topic_words):
            # np.argsort to sorting an array or a list or the matrix acc to their values
            sorted_topic_dist = np.argsort(topic_dist)

            # Next, to view the actual words present in those indexes we can make the use of the vocab created earlier
            topic_words = np.array(vocab_tf_idf)[sorted_topic_dist]

            # so using the sorted_topic_indexes we ar extracting the words from the vocabulary
            # obtaining topics + words
            # this topic_words variable contains the Topics  as well as the respective words present in those Topics
            topic_words = topic_words[:-n_top_words:-1]

            nv.append(topic_words)

        data= pd.DataFrame(nv)
        df=data.to_html()

        return render_template('lda.html',data=df)
import nltk
from sklearn import svm
@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = pd.read_csv("data.csv")
    tfidf = TfidfVectorizer(max_features=5000)
    X = data['reviews']
    y = data['Sentiment']

    X = tfidf.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = svm.SVC(kernel='linear', C=100, random_state=42)
    clf.fit(X_train, y_train)
    if request.method == 'POST':
        comment = request.form['comment']
        nltk.download('vader_lexicon')
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        score = ((sid.polarity_scores(str(comment))))['compound']

        if (score > 0):
            label = 'Positive 😊 😊'
        elif (score == 0):
            label = 'Negative 🙁 🙁'
        else:
            label = 'Neutral 😐 😐'

        li = []
        li.append(comment)
        df = pd.DataFrame(li, columns=['comment'])

        def remove_punctuations(text):
            for punctuation in string.punctuation:
                text = text.replace(punctuation, '')
            return text

        df['no_punc'] = df['comment'].apply(remove_punctuations)
        stop_words = set(stopwords.words('english'))
        df['tokenized'] = df['no_punc'].apply(word_tokenize)
        df['lower'] = df['tokenized'].apply(lambda x: [word.lower() for word in x])
        df['stopwords_removed'] = df['lower'].apply(lambda x: [word for word in x if word not in stop_words])

        df['pos_tags'] = df['stopwords_removed'].apply(nltk.tag.pos_tag)

        LR = r"""
              NP: {<NN><JJ>}
                  {<JJ><NN>}
                  {<VBP><NN>}
                  {<JJ><NN><NN>}
                  {<RB><JJ><NN>}
                  {<NN><NN><JJ>}
                  {<JJ><JJ><NN>}
                  {<RB><RB><NN>}
                  {<JJ><NN><JJ>}
                  {<VBP><JJ><NN>}
                  {<NN><RB><JJ>}
                  {<NN><JJ><NN>}
                  {<VBP><NNS>}
                  {<NNS><JJ>}
                  {<NNS><NN>}
                  {<NNP><JJ>}
                  {<NN><NNP>}
                  {<NNP><NN>}
            """
        nv = []
        cp = nltk.RegexpParser(LR)

        for i in df.index:
            tags = df['pos_tags'][i]
            result = cp.parse(tags)
            for j in result.subtrees():
                if j.label() == 'NP':
                    ner = nltk.ne_chunk(j)
                    nv.append(ner)
        data = pd.DataFrame(nv)
        stt = []
        for i in range(len(data)):
            stt.append(data[0][i][0])


        st = []
        for i in range(len(data)):
            st.append(data[1][i][0])


        a = []
        for i in range(len(stt)):
            a.append(f'{stt[i]} {st[i]}')

        d = pd.DataFrame(a, columns=['Reviews'])
        sentiment_list = []
        for row_num in range(len(d)):
            sentence = d['Reviews'][row_num]
            sentence = [sentence]
            vect = tfidf.transform(sentence).toarray()
            my_prediction = clf.predict(vect)
            if my_prediction == 1:
                sentiment_list.append("Positive 😊😊")

            elif my_prediction == -1:
                sentiment_list.append("Negative 🙁🙁")

            else:
                sentiment_list.append("Neutral")

        d['Sentiment'] = sentiment_list

        pos = 0
        neg = 0
        neut = 0
        for row_num in range(len(d)):
            if d['Sentiment'][row_num] == "Positive 😊 😊":
                pos = pos + 1
            elif d['Sentiment'][row_num] == "Negative 🙁 🙁":
                neg = neg + 1
            else:
                neut = neut + 1
    db = d.to_html()

    return render_template('result.html',prediction = label, data=db, pos=pos, neg=neg, neut=neut)

@app.route('/liste')
def liste():
  concaten = ""
  liste = []
  for parent, dnames, fnames in os.walk("static/files/"):
    for fname in fnames:
      filename = os.path.join(parent, fname)
      liste.append(filename)

      uploaded_df = pd.read_csv(filename)
      # Apply sentiment function to get sentiment score
      uploaded_df_sentiment = vader_sentiment_scores(uploaded_df)
      pos = 0
      neg = 0
      neut = 0
      num = 0
      for row_num in range(len(uploaded_df_sentiment)):
          num = num + 1
          if uploaded_df_sentiment['Sentiment'][row_num] == "Positive 😊 😊":
              pos = pos + 1
          elif uploaded_df_sentiment['Sentiment'][row_num] == "Negative 🙁 🙁":
              neg = neg + 1
          else:
              neut = neut + 1

      per_pos = (pos/num)*100
      per_neg = (neg/num) * 100
      per_neut = (neut/num) * 100
  return render_template('liste.html', liste=liste, pos=per_pos, neg=per_neg, neut=per_neut, num=num)



# end of code to run it
if __name__ == "__main__":
    app.run(debug=True)


