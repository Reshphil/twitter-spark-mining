from mongoengine import *
import json
import re

connect('twitterusers')

class User(Document):
    user_id = StringField(required=True, primary_key=True, unique=True)
    user_name = StringField(required=True, max_length=50)
    followers_count = IntField(required=True)
    following_count = IntField(required=True)
    listed_count = IntField(required=True)
    ratio = FloatField(required=True)
    tweet_count = IntField(required=True)
    tweets = ListField()
    # meta
    meta = {'allow_inheritance': True}

class Tweet(Document):
    tweet_id = StringField(required=True, primary_key=True, unique=True)
    user_id = StringField(required=True)
    tweet_text = StringField(required=True)
    raw_json_as_dict = DictField(required=True)

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

for user in users:
    print(user.user_id)
    print(user.username)
    print(user.followers_c)
    print(user.following_c)
    print(user.ratio)
    print(user.listed_c)
    print(user.tweets_c)
    # begin saving to DB
    tweet_id_array = []
    # populate user object
    user_object = User(user_id=user.user_id, \
        user_name=user.username, \
        followers_count=user.followers_c, \
        following_count=user.following_c, \
        listed_count=user.listed_c, \
        tweet_count=user.tweets_c, \
        ratio=user.ratio).save()
    # get tweet IDs for user
    tweets_ids = sqlContext.sql("SELECT distinct(id_str) as tweet_id \
        FROM tweets WHERE user.id_str LIKE "+user.user_id).collect()
    # build each tweet and save to DB
    for twid in tweets_ids:
        tweet_df = sqlContext.sql("SELECT * FROM tweets \
            WHERE id_str LIKE "+twid.tweet_id)
        tweet = tweet_df.first()
        json_to_be_loaded = tweet_df.toJSON().first()
        json_to_be_loaded = re.sub('[$]oid+', 'oid', json_to_be_loaded)
        print(json_to_be_loaded)
        json_to_dict = json.loads(json_to_be_loaded)
        tweet_object = Tweet(tweet_id=tweet.id_str, \
            user_id=tweet.user.id_str, \
            tweet_text=tweet.text, \
            raw_json_as_dict=json_to_dict).save()
        tweet_id_array.append(tweet_object.tweet_id)
        print(" --- : "+tweet.text)
    user_object.tweets = tweet_id_array
    user_object.save()
    # done saving to DB
    print("\n\n --- \n\n")
