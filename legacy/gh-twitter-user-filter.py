# import processing as twpr
# from imp import reload as reload # to use: twpr = reload(twpr)
# import re # needed for stripping text of special characters
# import unicodedata # needed for unicode to ASCII conversion without errors
# import gensim # needed for text clustering
# from gensim import corpora, models, similarities # needed for text clustering
# import sklearn
# import nltk
# nltk.download() # ensure all the necessary corpora are present for lemmatization

'''You might need to use python 3 with pyspark to get gensim, numpy and sklearn properly installed on OS X'''

# point to the folder where the tweets to be analyzed reside
fs_path = "/Users/timo/Code/spark/"



# for testing: path = "/Users/timo/Ruby/GetTweets/stored_tweets/2015-05-08.json"
#path = "/Users/timo/Ruby/GetTweets/stored_tweets/*"

# the rest of the path for where the tweets to be analyzed reside
path = fs_path+"stored_tweets/2015-05-08.json"
# load tweets into Apache SparkSQL (sqlContext)
tweets = sqlContext.read.json(path)

# register SparkSQL temptable called 'tweets'
tweets.registerTempTable("tweets")
# get the text content of each tweet
u = sqlContext.sql("SELECT distinct(user.id_str) as user_id, first(user.screen_name) as username, \
    min(user.followers_count) as followers_c, \
    max(user.friends_count) as following_c, \
    (min(user.followers_count)/max(user.friends_count)) as ratio, \
    min(user.listed_count) as listed_c, \
    max(user.statuses_count) as tweets_c \
    FROM tweets GROUP BY user.id_str \
    ORDER BY ratio DESC")

users = u.collect()

for user in users[0:5]:
    user_tweets = sqlContext.sql("SELECT distinct(id_str) as tweet_id, text as tweet_text \
        FROM tweets WHERE user.id_str LIKE "+user.user_id).collect()
    for u_tw in user_tweets:
        print(" --- : "+u_tw.tweet_text)
    print(user.user_id)
    print(user.username)
    print(user.followers_c)
    print(user.following_c)
    print(user.ratio)
    print(user.listed_c)
    print(user.tweets_c)
    print("\n\n --- \n\n")
