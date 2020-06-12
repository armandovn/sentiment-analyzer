import re, string
# Import stepwords 
from nltk.corpus import stopwords
# Import the normalizing tags and lemmatizer class
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
# Import the NaiveBayesClassifier to build the model
from nltk import classify
from nltk import NaiveBayesClassifier

class SentimentAnalysis:
    # Normalize data and remove all stopwords
    def removeHyperlinks(self, token):
        return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
    
    def removeTwitterUsers(self, token):
        return re.sub("(@[A-Za-z0-9_]+)","", token)

    def filterTokens(self, token_list, stop_words = stopwords.words('english')):
        clear_tokens = []
        for token in token_list:
            # Remove hyperlinks
            token = self.removeHyperlinks(token)
            # Remove twitter users
            token = self.removeTwitterUsers(token)

            # Filter the valid tokens
            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                clear_tokens.append(token.lower())

        return clear_tokens

    def normalizeTokens(self, token_list):
        tokens = []
        lemmatizer = WordNetLemmatizer()
        # We iterate thowout all tokens
        for token, tag in pos_tag(token_list):
            # We need to define the pos for lemmatizer, remember that we have the next tags for words:
            # NNP: Noun, proper, singular
            # NN: Noun, common, singular or mass
            # IN: Preposition or conjunction, subordinating
            # VBG: Verb, gerund or present participle
            # VBN: Verb, past participle
            # so we just need know the tow start letters to classified 
            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            # lemmatizer.lemmatize is use to normalize the data, we pass the pos (noun -> a, verb -> v, adjective -> a)
            tokens.append(lemmatizer.lemmatize(token, pos))

        return tokens

    # Get all words in the vocabulary
    # This method is necessary if we want to know thos most common word
    # on our data
    def getAllWords(self, cleaned_tokens_list):
        for tokens in cleaned_tokens_list:
            for token in tokens:
                yield token

    # Convert tokens to dictonary.
    # In this step we convert the array tokens into dictionary with token and value.
    # The value indicates if the token exist in the sentence, how we are iterate into the sentece
    # we can say that each token true value.
    # This is necesary to train the data model.
    def getPreparedTrainData(self, cleaned_tokens_list):
        for token_list in cleaned_tokens_list:
            yield dict([token, True] for token in token_list)

    def getCleanedAllPostTokens(self, array_tweet_tokens):
        clean_tokens = []
        for tokens in array_tweet_tokens:
            clean_tokens.append(self.normalizeTokens(self.filterTokens(tokens, stopwords.words('english'))))
        return clean_tokens
    
    def getCleanedPostTokens(self, token_list):
        return self.normalizeTokens(self.filterTokens(token_list, stopwords.words('english')))

    # Building and training our model
    def trainModel(self, train_data, test_data):
        return NaiveBayesClassifier.train(train_data)
