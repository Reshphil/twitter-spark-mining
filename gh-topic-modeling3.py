'''You might need to use python 3 with pyspark to get gensim, numpy and sklearn properly installed on OS X'''

fs_path = "/Users/timo/Code/spark/"

import processing as twpr
from imp import reload as reload # to use: twpr = reload(twpr)
import re # needed for stripping text of special characters
import unicodedata # needed for unicode to ASCII conversion without errors
import gensim # needed for text clustering
from gensim import corpora, models, similarities # needed for text clustering
import sklearn
import nltk
# nltk.download() # ensure all the necessary corpora are present for lemmatization

# for testing: path = "/Users/timo/Ruby/GetTweets/stored_tweets/2015-05-08.json"
#path = "/Users/timo/Ruby/GetTweets/stored_tweets/*"

path = fs_path+"stored_tweets/2015-05-08.json"
tweets = sqlContext.read.json(path)

tweets.registerTempTable("tweets")
tweet_texts = sqlContext.sql("SELECT text FROM tweets")
texts = twpr.run(tweet_texts)

dictionary = twpr.buildDictionaryFromTexts(texts)
corpus = twpr.buildCorpusFromDictionaryAndTexts(texts, dictionary)

num_topics = 25
tweet_ids = sqlContext.sql("SELECT id_str as id, text FROM tweets")
# done

distros = twpr.doLDA(corpus, dictionary, num_topics, tweet_ids)
distros_all = distros.collect()
# done

topics = twpr.TFIDFsFromTopicDistributions(distros_all[0:-1], sqlContext, corpus, dictionary)
twpr.writeTFIDFsToCSV(topics)
