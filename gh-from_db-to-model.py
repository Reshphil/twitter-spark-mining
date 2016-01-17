import processing as twpr
from imp import reload as reload # to use: twpr = reload(twpr)
import re # needed for stripping text of special characters
import unicodedata # needed for unicode to ASCII conversion without errors
import gensim # needed for text clustering
from gensim import corpora, models, similarities # needed for text clustering
import sklearn
import nltk
# nltk.download() # ensure all the necessary corpora are present for lemmatization


# Use MongoDB to fetch the top 10 users with the highest ratio_per_tweet index
users = twpr.User.objects().order_by('ratio_per_tweet').limit(100)
# from those users, get the tweet texts
tweet_texts = twpr.User.getTextsFromUsers(users)
# and then pre-process those texts
texts = twpr.runWithoutMap(tweet_texts)
# build dictionary and corpus
dictionary = twpr.buildDictionaryFromTexts(texts)
corpus = twpr.buildCorpusFromDictionaryAndTexts(texts, dictionary)

# set LDA topic count parameter
num_topics = 25
# and do the LDA modeling
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, update_every=0, passes=20)

# get the distributions for each tweet
tweets_for_lda = twpr.User.getTweetsFromUsers(users)
distros = twpr.distrosForTweetsFromLDAModel(lda, dictionary, tweets_for_lda)

# print the topic keywords with the TF-IDF frequencies as weights
topics = twpr.TFIDFsFromMongoDBTopicDistributions(distros, corpus, dictionary)
twpr.writeTFIDFsToCSV(topics, 'new_tfidf.csv')
