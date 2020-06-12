import random
import pandas as pd
from flask import Flask, render_template, url_for, flash, redirect, request
from sentimentAnalysis import SentimentAnalysis
from twitterSearch import TwitterSearch
# Iimport the dataset for trin the model
from nltk.corpus import twitter_samples
# Import the class FreqDist to identyfied the most common words
from nltk import FreqDist
# Import sklearn for clasification reports
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Import sklearn.tokenize to tokenize the strings
from nltk.tokenize import word_tokenize

# Create the flask instace to execute the page
app = Flask(__name__)
app.config['SECRET_KEY'] = 'f898a7a6b03489d08a37f46aac83b568'

# Create the instance for sentiment Analisis
sentiment_analysis = SentimentAnalysis()

# Create a TwitterSearch instance
twitterSearch = TwitterSearch()
twitterSearch.twitterAuth('5U9PIoRNb5k4hEOoqUgNsRYEb', 'DXMZPPZGgk2y2eeIpG7FWhRzVGzPGQBL3Swcp0hpqDv0d4LF3o', '1054216895258230784-qzpEYyQEqYeVAPTR4wfjUqtyJK5gsy', 'Bg8D2DagusdlQ9HjIWg1utTRJmuymXj8T6Grd9AdF28vS')

positive_cleaned_tokens_list = sentiment_analysis.getCleanedAllPostTokens(twitter_samples.tokenized('positive_tweets.json'))
negative_cleaned_tokens_list = sentiment_analysis.getCleanedAllPostTokens(twitter_samples.tokenized('negative_tweets.json'))

# Formating the data
positive_tokens_for_model = sentiment_analysis.getPreparedTrainData(positive_cleaned_tokens_list)
negative_tokens_for_model = sentiment_analysis.getPreparedTrainData(negative_cleaned_tokens_list)

# Splitting the dataset for training and testing the model
positive_dataset = [(tweet_dict, "Positive")
                    for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                    for tweet_dict in negative_tokens_for_model]

# Combaninig all data
dataset = positive_dataset + negative_dataset

# Stirring the data
random.shuffle(dataset)

# The code splits the shuffled data into a ratio of 70:30 for training and testing, respectively.
train_data = dataset[:7000]
test_data = dataset[7000:]

classifier = sentiment_analysis.trainModel(train_data, test_data)

predict_frame = []
for expression in test_data:
    predict_frame.append([expression[1], classifier.classify(dict([token, True] for token in expression[0]))])

df = pd.DataFrame(predict_frame, columns=['Real','Predited'])

accuracy_score = accuracy_score(df['Real'],df['Predited'])
print(type(accuracy_score))
classification_report = classification_report(df['Real'],df['Predited'])
print(type(classification_report))
confusion_matrix = confusion_matrix(df['Real'],df['Predited'])
print(type(confusion_matrix))
all_words = sentiment_analysis.getAllWords(positive_cleaned_tokens_list + negative_cleaned_tokens_list)
freq_dist_pos = FreqDist(all_words)
common_words = freq_dist_pos.most_common(20)
print(type(common_words))

@app.route("/")
@app.route("/home")
def home():
  return render_template('home.html', accuracy_score=accuracy_score, classification_report=classification_report, confusion_matrix=confusion_matrix, common_words=common_words)

@app.route("/options")
def options():
  return render_template('options.html', title='Analisis options')

@app.route("/result_review", methods=['GET','POST'])
def result_review():
  if request.method == 'POST':
    review = request.form['review']
    review_tokens = word_tokenize(review)
    review_filter_tokens = sentiment_analysis.filterTokens(review_tokens)
    review_normilze_tokens = sentiment_analysis.normalizeTokens(review_filter_tokens)
    clasification_result = classifier.classify(dict([token, True] for token in review_normilze_tokens))
    return render_template('result_review.html', title='Review Analisis', review=review, review_tokens=review_tokens, review_filter_tokens=review_filter_tokens, review_normilze_tokens=review_normilze_tokens, clasification_result=clasification_result)
  
@app.route("/twitter_search", methods=['GET','POST'])
def twitter_search():
  if request.method == 'POST':
    topic = request.form['topic']
    tweets = twitterSearch.makeASearch(topic)
    return render_template('twitter_search.html', title='Twitter Search', tweets=tweets)

@app.route("/about")
def about():
  return render_template('about_us.html', title='About')

if __name__ == '__main__':
  app.run()