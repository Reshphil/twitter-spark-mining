from mongoengine import *
import json
import re
from pprint import pprint as pp

connect('twitterusers')

class Tweet(Document):
    tweet_id = StringField(required=True, primary_key=True, unique=True)
    user_id = StringField(required=True)
    tweet_text = StringField(required=True)
    raw_json_as_dict = DictField(required=True)

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
    # methods
    def getTweets(self):
        tweets = Tweet.objects(user_id=self.id)
        return tweets
    def getTweetsFromUsers(users):
        out_tweets = []
        for u in users:
            if isinstance(u, User):
                tweets = u.getTweets()
                for t in tweets:
                    out_tweets.append(t)
                return out_tweets
            else:
                raise Exception("User object is not instance of User class!")
    def getTextsFromUsers(users):
        texts = []
        for u in users:
            if isinstance(u, User):
                tweets = u.getTweets()
                for t in tweets:
                    texts.append(t.tweet_text)
                return texts
            else:
                raise Exception("User object is not instance of User class!")
