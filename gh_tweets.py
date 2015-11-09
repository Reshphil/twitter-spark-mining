path = '/stored_tweets/*.json'


tweets = sqlContext.read.json(path)

# tweets.printSchema()
tweets.registerTempTable("tweets")

favs = sqlContext.sql("SELECT retweeted_status.id_str FROM tweets")

retweets_count = favs.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
sorted_retweets = sorted(retweets_count.collect(), key=lambda word: word[1], reverse=True)[0:9]

ids = []
for i in sorted_retweets:
    if (i[0][0] is not None):
        query = "SELECT text FROM tweets WHERE id_str = "+str(i[0][0])
        t = sqlContext.sql(query).take(1)
        print t
        tweet_text = str(t)
        print tweet_text
        str(ids.append({'tweet_id': i[0], 'retweets': i[1], 'text': tweet_text}))



top_retweets = sqlContext.sql("SELECT text FROM tweets WHERE id_str IN ("+sorted_retweets.+")")
