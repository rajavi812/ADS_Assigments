import numpy as np
import pandas as pd
from textblob import TextBlob
from nltk.stem import PorterStemmer
import nltk.tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')




# Load dataset
dataset = pd.read_csv('reviews.csv')
print(dataset.head())

print(dataset.describe())
print(dataset.info())

# splitting rating column to keep the rating and excluding the remaining
dataset['Rating'] = dataset['Rating'].apply(lambda x: x.split()[0])
dataset['Rating'] = pd.to_numeric(dataset['Rating'])

print(dataset.head())
print(dataset.columns)

dataset['polarity'] = [TextBlob(sentence).sentiment.polarity for sentence in dataset['Comment']]
dataset['subjectivity'] = [TextBlob(sentence).sentiment.subjectivity for sentence in dataset['Comment']]
dataset['word_list'] = [TextBlob(sentence).words.lemmatize() for sentence in dataset['Comment']]
print(dataset.head(10))

# words in a comment, comment-wise
print(dataset['word_list'][0:5])
print("Columns in the dataset are: ", dataset.columns)

# making a tupple
doc = [(text, star) for text, star in zip(dataset['Comment'], dataset['Rating'])]

# Cleaning the dataset:
# Removing stopwords: common words that do not add any sentiment (a, the, in, etc)
# Stemming: Converting words to their root form
# Replacing punctuation marks with a blank
# Converting Uppercase to Lowercase

stop = set(stopwords.words('english'))
stemmer = PorterStemmer()
wordlist = []

for text, star in doc:
    for word in text.split():
        w = word.lower().replace(',', '').replace('!', '').replace('.', '')
        # making a list of the most common words from the list after stemming
        wordlist.append(stemmer.stem(w))

# Finding the frequency of all the words in the wordlist
wordlist = nltk.FreqDist(wordlist)

print("Wordlist: ", wordlist)

# Making a list of the common words
featureList = [word for (word, _) in list(wordlist.most_common(4000))]
print("FeatureList: ", featureList)


# Creating a dictionary of the words
def search_feature(txt):
    words = set(txt.split())
    feature = {}
    for wd in words:
        if wd not in stop:
            wd = wd.lower().replace('!', '').replace(',', '').replace('.', '').replace('?', '')
            wd = stemmer.stem(wd)
            feature[wd] = (wd in featureList)
    print(feature)
    return feature


featureSets = [(search_feature(text), star) for (text, star) in doc]

print('FeatureSets', featureSets[1])

temp = []
positive = 0
negative = 0
# Labeling comments as positive or negative
for i in range(len(featureSets)):
    if featureSets[i][1] >= 4:
        temp.append((featureSets[i][0], 'positive'))
        positive += 1
    else:
        temp.append((featureSets[i][0], 'negative'))
        negative += 1


print("Positive: ", positive)
print("Negative: ", negative)

# lemmatization: Better form of stemming, ie, covert to base form but in a way that sentiments can be understood
# from the word

trainingSet = temp[:4001]
testingSet = temp[4001:]

# Implementing NaiveBayes algorithm

nb = nltk.NaiveBayesClassifier.train(trainingSet)
print("Naive Bayes Classifier accuracy: ", nltk.classify.accuracy(nb, testingSet)*100)
nb.show_most_informative_features(25)


# Implementing LogisticRegression algorithm
lr = SklearnClassifier(LogisticRegression())
lr.train(trainingSet)
print("Logistic Regression classifier accuracy percent:", (nltk.classify.accuracy(lr, testingSet))*100)


X = dataset['Comment']
y = dataset['Rating']
# print(y)

lem = WordNetLemmatizer()

def own_analyser(phrase):
    phrase = phrase.split()
    for i in range(0,len(phrase)):
        k = phrase.pop(0)
        if k not in (string.punctuation and stop):
            phrase.append(lem.lemmatize(k).lower())
        # print("Phrase: ", phrase)
    return phrase


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
pipeline = Pipeline([
    ('BOW', CountVectorizer(analyzer=own_analyser)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print('Accuracy from pipeline:', accuracy_score(y_test,y_pred)*100)


