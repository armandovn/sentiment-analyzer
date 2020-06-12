#Se importa la librería tweepy
import tweepy

class TwitterSearch:
    def twitterAuth(self, consumer_key, consumer_secret, access_token, access_token_secret):
        # Twitter Authentication
        self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(self.auth)

    def getUserName(self):
        # Return the auth Twitter User
        return api.me().name

    def makeASearch(self, topic_search="covid-19", tweets_number=10, leguanje="en"):
        tweets = []
        for tweet in tweepy.Cursor(self.api.search, topic_search, tweet_mode="extended", lang=leguanje).items(tweets_number):
            if(len(tweet._json['full_text']) > 30): tweets.append(tweet._json['full_text'] + '\n')
        return tweets