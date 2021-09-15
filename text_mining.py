import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import time

#time for calculation of runtime
StartTime = time.time()

#loads in corona data from csv file
try:
    df = pd.read_csv('data/text_data/Corona_NLP_train.csv', na_filter=False, encoding='latin-1' )
except Exception as x:
    print('error> reading data file' + str( x ))
    sys.exit()

##QUESTION 1.1
#prints the possible sentiments of a tweet
print("\nThe possible tweet's sentiment is:", df.Sentiment.unique(), "\n")

#prints the total number of each sentiment to view the 2nd most popular sentiment
print("The second most popular sentiment is:")
print(df["Sentiment"].value_counts().iloc[[1]], "\n")

#calculates the number of extremely positive sentiments per date, and prints the larges number with that corresponding date
df_extreme_pos = df.loc[df["Sentiment"] == "Extremely Positive"]
print("Date with most extremely positive tweets:")
print(df_extreme_pos["TweetAt"].value_counts().nlargest(1), "\n")

#removes the links in the tweets as they are irrelevant
df['OriginalTweet'] = df['OriginalTweet'].str.replace(r'http\S+', '',regex=True)

#converts message to lower case
df['OriginalTweet'] = df['OriginalTweet'].str.lower()

#replaces non-alphabetical characters with whitespace
df['OriginalTweet'] = df['OriginalTweet'].str.replace(r'[^a-z]', ' ', regex=True)

#strips extra whitespaces to have only a single whitespace between characters 
df['OriginalTweet'] = df['OriginalTweet'].str.strip()


##QUESTION 1.2
#tokenize the tweets
df['OriginalTweet'] = df['OriginalTweet'].apply(word_tokenize)

#prints the total number of all words with repetition
totalBefore = pd.Series(np.concatenate(df.OriginalTweet)).value_counts().sum()
print("Total number of all words with repetitions:" , totalBefore, "\n")

#prints the number of distinct (unique) words
totalDistinct = pd.Series(np.concatenate(df.OriginalTweet)).nunique()
print("Number of distinct words:" , totalDistinct, "\n")

#prints the 10 most frequent words
print("Top 10 most frequent words:")
print(pd.Series(np.concatenate(df.OriginalTweet)).value_counts().nlargest(10), "\n")

#removes stop words
#uses approach from Liam Forley at https://stackoverflow.com/questions/29523254/python-remove-stop-words-from-pandas-dataframe
stop = stopwords.words('english')
df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: [item for item in x if item not in stop])

#removes words with <= 2 characters
df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: [item for item in x if len(item)>2])

#recalculate the count of words with repetition and prints 10 most frequent after removal of stopwords and small words
totalAfter = pd.Series(np.concatenate(df.OriginalTweet)).value_counts().sum()
print("Total number of all words with repetitions after removal of stopwords and <=2 chars:" , totalAfter, "\n")

print("Top 10 most frequent words after removal:")
print(pd.Series(np.concatenate(df.OriginalTweet)).value_counts().nlargest(10), "\n")


##QUESTION 1.3
#plot a histogram as a line chart with word frequencies
#uses list() approach from Mike Muller at #https://stackoverflow.com/questions/35523635/extract-values-in-pandas-value-counts
#and uses ascending and normalize approach from Anna Zverkova at https://re-thought.com/pandas-value_counts/
Y = pd.Series(np.concatenate(df.OriginalTweet)).value_counts(ascending=True, normalize=True).tolist()
totalWords = pd.Series(np.concatenate(df.OriginalTweet)).nunique() + 1
X = list(range(1,totalWords))
plt.plot(X,Y)
plt.title( 'Word frequencies Histogram' )
plt.savefig( 'outputs/q1.3.png' )

##QUESTION 1.4
#loads in original data to dataframe
try:
    dfQ4 = pd.read_csv('data/text_data/Corona_NLP_train.csv', na_filter=False, encoding='latin-1' )
except Exception as x:
    print('error> reading data file' + str( x ))
    sys.exit()

#stores tweet in numpy array
corpus = np.array(dfQ4['OriginalTweet'])
classification = np.array(dfQ4['Sentiment'])

#produce sparse representation of term-document matrix with a countVectorizer - https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
vec = CountVectorizer()
X = vec.fit_transform(corpus)
#print(vec.vocabulary_)

#multinomial naive bayes classifier
clf = MultinomialNB()
clf.fit(X, classification)
predictions = clf.predict(X)
print('The confusion matrix for the Multinomial Naive Bayes classifier:\n' , confusion_matrix(classification, predictions))
accuracy = accuracy_score(classification, predictions)
error_rate = 1 - accuracy
print('The error rate is:' , error_rate)

#Calculation of runtime
EndTime = time.time()
Runtime = EndTime - StartTime
print("\nTotal runtime is:", Runtime)