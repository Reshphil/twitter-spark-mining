'''You might need to use python 3 with pyspark to get gensim, numpy and sklearn properly installed on OS X'''

# point to the folder where the tweets to be analyzed reside
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

# the rest of the path for where the tweets to be analyzed reside
path = fs_path+"stored_tweets/2015-05-08.json"
# load tweets into Apache SparkSQL (sqlContext)
tweets = sqlContext.read.json(path)

# register SparkSQL temptable called 'tweets'
tweets.registerTempTable("tweets")
# get the text content of each tweet
tweet_texts = sqlContext.sql("SELECT text FROM tweets")

# run the processing .run() in the processing.py for the texts
# as output, we have pre-processed texts ready for gensim dictionary and gensim building
texts = twpr.run(tweet_texts)

# build Gensim dictionary and corpus with helper methods in processing.py
dictionary = twpr.buildDictionaryFromTexts(texts)
corpus = twpr.buildCorpusFromDictionaryAndTexts(texts, dictionary)

# set LDA topic count parameter
num_topics = 25
# in order to map LDA output and actual tweets for further analysis, select tweet IDs and texts
tweet_ids = sqlContext.sql("SELECT id_str as id, text FROM tweets")
# now we have all necesssary pre-processed data for LDA analysis

# use the pre-processed inputs to do the LDA analysis
distros = twpr.doLDA(corpus, dictionary, num_topics, tweet_ids)

# now we have the Apache Spark RDD object we can either .take(5) or .collect() all
distros_all = distros.collect()
# now we have the LDA topic probability distributions in memory

hdp = twpr.doHDP(corpus, dictionary)

# to make sense of the LDA output, we need to somehow look at the data
# thus, we'll write the topics into CSV, weighted with TF-IDF frequencies
topics = twpr.TFIDFsFromTopicDistributions(distros_all[0:-1], sqlContext, corpus, dictionary)
twpr.writeTFIDFsToCSV(topics)
# processing done
