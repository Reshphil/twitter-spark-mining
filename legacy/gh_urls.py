path = '/Users/timo/Code/GetTweets/stored_tweets/*.json'

tweets = sqlContext.read.json(path)

# tweets.printSchema()
tweets.registerTempTable("tweets")

u = sqlContext.sql("SELECT entities.urls FROM tweets")
#u.flatMap(lambda tweet: tweet[0]).map(lambda url: url.expanded_url).take(1)


a = u.flatMap(lambda tweet: tweet[0]).map(lambda url: (url.expanded_url, 1))
a.reduceByKey(lambda a, b: a+b)

# u.flatMap(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
#retweets_count = favs.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)

sorted_urls = sorted(a.reduceByKey(lambda a, b: a+b).collect(), key=lambda word: word[1], reverse=True)[0:25]
# done prep






ids = []
for i in sorted_urls:
    if (i[0] is not None) and (len(i[0]) > 0):
        if (i[0][0] is not None) and (len(i[0][0]) > 0):
            query = "SELECT text FROM tweets WHERE id_str = "+str(i[0][0])
            a = sqlContext.sql(query).take(1)
            if (a is not None) and (len(a) > 0):
                if (a[0] is not None) and (len(a[0]) > 0):
                    if (a[0].text is not None) and (len(a[0].text) > 0) and (len(i[0].id_str) > 0):
                        t = a[0].text.encode('ascii', 'ignore')
                        tweet_text = str(t)
                        id_str = str(i[0].id_str)
                        if (len(tweet_text) > 0) and (len(id_str) > 0):
                            tweet_d = {'tweet_id': id_str, 'retweets': i[1], 'text': tweet_text}
                            print tweet_d
                            str(ids.append(tweet_d))
# yes

import csv

def write_csv():
    with open('/Users/timo/Code/top_retweets.csv', 'wb') as csvfile:
        rtwriter = csv.writer(csvfile, delimiter=';', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        # write header first:
        rtwriter.writerow(['tweet_id', 'retweet_count', 'orig_tweet_text'])
        print "Headers written. Starting to write rows."
        # then write rows:
        for i in ids:
            # also remember tho handle the \n's, ;'s and all that jazz
            row = [i["tweet_id"], i["retweets"], i["text"].replace(";", "").rstrip('\n').replace("\n", " ")]
            rtwriter.writerow(row)
        print "Writing done."
#done

write_csv()

#c = sqlContext.sql("SELECT text FROM tweets WHERE id_str = 598316479990337536").take(1)
#top_retweets = sqlContext.sql("SELECT text FROM tweets WHERE id_str IN ("+sorted_retweets.+")")
# done top
