from mongoengine import *
import json
import re
from pprint import pprint as pp

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
    ratio_per_tweet = FloatField()
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
t = sqlContext.sql("SELECT distinct(user.id_str) as user_id, first(user.screen_name) as username, \
    min(user.followers_count) as followers_c, \
    max(user.friends_count) as following_c, \
    (min(user.followers_count)/max(user.friends_count)) as ratio, \
    min(user.listed_count) as listed_c, \
    max(user.statuses_count) as tweets_c \
    FROM tweets \
    GROUP BY user.id_str ORDER BY ratio DESC \
    LIMIT 1000")

t.registerTempTable("tweet_users")

u = sqlContext.sql("SELECT * FROM tweet_users WHERE following_c > 0 AND ratio > 0 AND followers_c > 250 AND listed_c > 5 AND tweets_c > 100 LIMIT 1000")

users = u.collect()

def printUserInfo(user):
    # print(user.user_id)
    print(user.username)
    # print(user.followers_c)
    # print(user.following_c)
    # print(user.ratio)
    # print(user.listed_c)
    # print(user.tweets_c)

def proceedToSaveTweets(user, tweet_id_array):
    # get tweet IDs for user
    tweets_ids = sqlContext.sql("SELECT distinct(id_str) as tweet_id \
        FROM tweets WHERE user.id_str LIKE "+user.user_id).collect()
    # build each tweet and save to DB
    try:
        # increment the counter for tweet_counter
        tweet_counter = 0
        for twid in tweets_ids:
            tweet_counter = tweet_counter+1
            print("Processing tweet number ", str(tweet_counter), " of ", \
                str(len(tweets_ids)), " tweets total.")
            try:
                tweet_df = sqlContext.sql("SELECT * FROM tweets \
                    WHERE id_str LIKE "+twid.tweet_id)
                tweet = tweet_df.first()
                json_to_be_loaded = tweet_df.toJSON().first()
                json_to_be_loaded = re.sub('[$]oid+', 'oid', json_to_be_loaded)
                # print(json_to_be_loaded)
                json_to_dict = json.loads(json_to_be_loaded)
            except:
                print("Loading tweet from Spark failed because: ", e)
                raise
            else:
                # try to save Tweet
                try:
                    tweet_object = Tweet(tweet_id=tweet.id_str, \
                        user_id=tweet.user.id_str, \
                        tweet_text=tweet.text, \
                        raw_json_as_dict=json_to_dict).save()
                    tweet_id_array.append(tweet_object.tweet_id)
                    # print(" --- : "+tweet.text)
                except ValidationError as e:
                    print("FYI: Saving tweet ", tweet.text, " failed because: ", e)
                    pass
    except:
        print("Getting Tweets for User Failed!")
        raise
    else:
        return tweet_id_array

def saveUser(user, ratio):
    # populate user object
    try:
        user_object = User(user_id=user.user_id, \
            user_name=user.username, \
            followers_count=user.followers_c, \
            following_count=user.following_c, \
            listed_count=user.listed_c, \
            tweet_count=user.tweets_c, \
            ratio=ratio).save()
    except ValidationError as e:
        print("FYI: Saving user ", user.username, " failed because: ", e)
        pass
    else:
        tweet_id_array = []
        tweet_id_array = proceedToSaveTweets(user, tweet_id_array)
        try:
            user_object.tweets = tweet_id_array
            user_object.save()
        except ValidationError as e:
            print("FYI: Saving user with tweets ", user.username, " failed because: ", e)
            pass

def processUsers(users):
    counter = 0
    for user in users:
        # start by incrementing the counter
        counter = counter+1
        print("Processing user ", str(counter), " out of ", str(len(users)), "users total.")
        # begin saving to DB
        # since for some users the ratio can be None, do a check for that
        if user.ratio is not None:
            ratio = user.ratio
        else:
            ratio = -1
        # try:
        saveUser(user, ratio)
        # except:
        #     print("Saving user failed because..")
        #     pass

processUsers(users)

def reCalculateRatioForUsers():
    for p in User.objects():
        if p.following_count is not 0 and p.following_count is not -1:
            p.ratio = float(float(p.followers_count)/float(p.following_count))
        else:
            p.ratio = 0.00
        p.save()

def printUsersFromDB():
    for p in User.objects().order_by('ratio_per_tweet'):
        print("\n\n ---- ")
        print("Username: ", str(p.user_name))
        print("Ratio: ", str(p.ratio))
        print("Ratio per tweet: ", str(p.ratio_per_tweet))
        print("Following: ", str(p.following_count))
        print("Followers: ", str(p.followers_count))
        print("Tweets in this dataset: ", str(len(p.tweets)))

def calculateRatioPerTweet():
    for p in User.objects():
        if p.ratio is not 0 and p.ratio is not -1:
            rpt = p.ratio/len(p.tweets)
            p.ratio_per_tweet = rpt
        p.save()
