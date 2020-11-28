import numpy as np
import pandas as pd
import nltk
import string
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from textblob import TextBlob
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

# Load the dataset
dataset = pd.read_csv('reviews.csv')

# Checking the contents of the dataset
print('Dataset size:', dataset.shape)
print('Columns are:', dataset.columns)
print(dataset.head())

dataset['polarity'] = [TextBlob(sentence).sentiment.polarity for sentence in dataset['Comment']]
dataset['subjectivity'] = [TextBlob(sentence).sentiment.subjectivity for sentence in dataset['Comment']]

# Removing extra content from the rating column and keeping only the relevant part - rating
print('Splitting the Ratings column to remove the extra content')
dataset[["Rating", "out", "of", "5", "stars"]] = dataset["Rating"].str.split(" ", expand = True)
print('Columns are:', dataset.columns)
dataset = dataset.drop(dataset.columns[[2, 3, 4, 5, 6, 7]], axis=1)
dataset['Rating'] = pd.to_numeric(dataset['Rating'])
print(dataset.head())
print('Dataset size:', dataset.shape)
print('Columns are:', dataset.columns)
print(dataset.info())

# Converting all text to lowercase
print('Convert text to lower case')
dataset['Comment'] = dataset['Comment'].str.lower()
print(dataset.head())

# Removing the punctuation marks
print('Remove Punctuation')
print(string.punctuation)


def remove_punc(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text


dataset['Comment'] = dataset['Comment'].apply(lambda x: remove_punc(x))
print(dataset.head())

# Removing the numbers from the text
print('Removing Numbers from the text')
dataset['Comment'] = dataset['Comment'].str.replace('\d+', '')
print(dataset.head())


# Tokenizing the words (Separating words from a sentence)
print('Tokenization of words')
comments = dataset['Comment']
dataset['Comment'] = comments.apply(word_tokenize)
print(dataset.head())

# Removing unnecessary words - stopwords
print('Removing stopwords')
stopword = nltk.corpus.stopwords.words('english')
print(stopword)


def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text


dataset['Comment'] = dataset['Comment'].apply(lambda x: remove_stopwords(x))
print(dataset.head())


# Performing Stemming
print('Performing Stemming')
st = nltk.PorterStemmer()


def stemming(text):
    text = [st.stem(word) for word in text]
    return text


dataset['Comment'] = dataset['Comment'].apply(lambda x: stemming(x))
print(dataset.head())


# Performing Lemmatization
# observed that when I performed only lemmatization, the words did not get stemmed, hence implementing both
print('Performing Lemmatization')
wn = nltk.WordNetLemmatizer()


def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text


dataset['Comment'] = dataset['Comment'].apply(lambda x: lemmatizer(x))
print(dataset.head())

# Forming a sentence from the remaining words
print('Joining the words to form a string/sentence')
corpus = []
for i in range(0, 5000):
    review = ' '.join(dataset['Comment'][i])
    corpus.append(review)

print("Corpus", corpus)


# Classifying Reviews as positive/negative/neutral based on ratings
print("Classifying Reviews as positive/negative/neutral based on ratings")
classify = []
for i in range(0, 5000):
    if dataset['Rating'][i] > 3.0: # Posiitve reviews
        classify.append(1)
    elif dataset['Rating'][i] < 3.0: # Negative reviews
        classify.append(-1)
    else:
        classify.append(0) # Neutral reviews

print(classify)
dataset['Classify'] = classify

print(dataset.head())

# Vectorizing the words and counting the frequency for each
# Extracting the features
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 2].values
print(cv.vocabulary_)

print(X.shape)
print(y.shape)

X_train = X[0:3001, :]
print("X_train: ", X_train)

X_test = X[3001:4501, :]
print("X_test: ", X_test)

y_train = y[0:3001]
print("y_train", y_train)

y_test = y[3001:4501]
print("y_test", y_test)

X_val = X[4501:5001, :]
print("X_val", X_val)

y_val = y[4501:5001]
print("y_val", y_val)


print("***************LOGISTIC REGRESSION*****************")
lr = LogisticRegression(max_iter=5000) # Keeping it 5000 as there are 5000 rows
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred_lr)
print(cm)
acc_lr = accuracy_score(y_test, y_pred_lr)*100
print('Accuracy from Logistic Regression classifier:', acc_lr)
print(classification_report(y_test, y_pred_lr))

print("***************XGBOOST*****************")

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred_xg = xgb.predict(X_test)
cm = confusion_matrix(y_test, y_pred_xg)
print(cm)
acc_xgb = accuracy_score(y_test, y_pred_xg)*100
print('Accuracy from XGBoost:', acc_xgb)
print(classification_report(y_test, y_pred_xg))

print("***************RANDOM FOREST CLASSIFIER*****************")

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)
acc_rf = accuracy_score(y_test, y_pred_rf)*100
print('Accuracy from RandomForestClassifier:', acc_rf)

print("***************K-NEAREST NEIGHBOURS*****************")
# from sklearn.model_selection import GridSearchCV
# # Incase  of classifier like  knn the parameter to be tuned is n_neighbors
# param_grid = {'n_neighbors': np.arange(1, 10)}
# knn = KNeighborsClassifier()
# knn_cv = GridSearchCV(knn, param_grid, cv=5)
# print(knn_cv.fit(X_train, y_train))
# print(knn_cv.best_score_*100)
# i = knn_cv.best_params_
# print("i = " + str(i))
# # try to find best k value
error = []

for i in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    err = knn.score(X_test, y_test)
    error.append(err)
    print("For " + str(i) + " Neighbours, error is: " + str(err))

i = error.index(min(error))
i += 1
print("Minimum error is for " + str(i) + " Neighbours")

classifier = KNeighborsClassifier(n_neighbors = i)
classifier.fit(X_train,y_train)
y_pred_knn = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred_knn)
print(cm)
acc_knn = accuracy_score(y_test, y_pred_knn)*100
print('Accuracy from KNN:', acc_knn)
print(classification_report(y_test, y_pred_knn))

print("***************MULTINOMIAL NAIVE BAYESIAN CLASSIFIER*****************")

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)
cm = confusion_matrix(y_test, y_pred_mnb)
print(cm)
acc_mnb = accuracy_score(y_test, y_pred_mnb)*100
print('Accuracy from Naive Bayes:', acc_mnb)
print(classification_report(y_test, y_pred_mnb))

print("***********ACCURACY SCORE FOR TESTING SET***************")

table_lr = pd.DataFrame(data=[["Logistic Regression", acc_lr]], columns=['Model', 'Testing Accuracy %'])

table_xgb = pd.DataFrame(data=[["XGBoost", acc_xgb]], columns=['Model', 'Testing Accuracy %'])
table_xgb = table_xgb.append(table_lr, ignore_index=True)

table_rf = pd.DataFrame(data=[["Random Forest Classifier", acc_rf]], columns=['Model', 'Testing Accuracy %'])
table_rf = table_rf.append(table_xgb, ignore_index=True)

table_knn = pd.DataFrame(data=[["K Nearest Neighbour", acc_knn]], columns=['Model', 'Testing Accuracy %'])
table_knn = table_knn.append(table_rf, ignore_index=True)

table_mnb = pd.DataFrame(data=[["Multinomial Naive Bayes Classifier", acc_mnb]], columns=['Model', 'Testing Accuracy %'])
table_mnb = table_mnb.append(table_knn, ignore_index=True)


print(table_mnb)

print("********************VALIDATION SET********************")
print("Calculating accuracy scores for validation set......")
val_lr = lr.predict(X_val)
val_xgb = xgb.predict(X_val)
val_rf = rf.predict(X_val)
val_knn = knn.predict(X_val)
val_mnb = mnb.predict(X_val)


accuracy_lr = accuracy_score(y_val, val_lr)*100
accuracy_xgb = accuracy_score(y_val, val_xgb)*100
accuracy_rf = accuracy_score(y_val, val_rf)*100
accuracy_knn = accuracy_score(y_val, val_knn)*100
accuracy_mnb = accuracy_score(y_val, val_mnb)*100

print("***********ACCURACY SCORE FOR VALIDATION SET***************")

tLR = pd.DataFrame(data=[["Logistic Regression", accuracy_lr]], columns=['Model', 'Testing Accuracy %'])

tXGB = pd.DataFrame(data=[["XGBoost", accuracy_xgb]], columns=['Model', 'Testing Accuracy %'])
tXGB = tXGB.append(tLR, ignore_index=True)

tRF = pd.DataFrame(data=[["Random Forest Classifier", accuracy_rf]], columns=['Model', 'Testing Accuracy %'])
tRF = tRF.append(tXGB, ignore_index=True)

tKNN = pd.DataFrame(data=[["K Nearest Neighbour", accuracy_knn]], columns=['Model', 'Testing Accuracy %'])
tKNN = tKNN.append(tRF, ignore_index=True)

tMNB = pd.DataFrame(data=[["Multinomial Naive Bayes Classifier", accuracy_mnb]], columns=['Model', 'Testing Accuracy %'])
tMNB = tMNB.append(tKNN, ignore_index=True)

print(tMNB)