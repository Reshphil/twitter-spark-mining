'''You might need to use python 3 with pyspark to get gensim, numpy and sklearn properly installed on OS X'''

fs_path = "/Users/timo/Code/GetTweets/"

import processing as twpr
from imp import reload as reload # to use: twpr = reload(twpr)
import re # needed for stripping text of special characters
import unicodedata # needed for unicode to ASCII conversion without errors
import gensim # needed for text clustering
from gensim import corpora, models, similarities # needed for text clustering
import sklearn

# for testing: path = "/Users/timo/Ruby/GetTweets/stored_tweets/2015-05-08.json"
#path = "/Users/timo/Ruby/GetTweets/stored_tweets/*"

path = fs_path+"stored_tweets/2015-05-08.json"
tweets = sqlContext.read.json(path)
tweets.registerTempTable("tweets")
# tweets without stopwords
tweet_texts = sqlContext.sql("SELECT text FROM tweets")

texts = twpr.run(tweet_texts)

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
num_topics = 25

# above this is common to both using LDA directly from Gensim and running the proprietary functions below
# lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, update_every=0, passes=20)


tweet_ids = sqlContext.sql("SELECT id_str as id, text FROM tweets")

distros = twpr.doLDA(corpus, dictionary, num_topics, tweet_ids)
distros_all = distros.collect()

# import numpy
# Apache PySpark RDD.takeSample requires numpy
# distros.takeSample(False, 100) # distros.collect()
# done

topics = twpr.wordCountFromTopicDistributions(distros_all, sqlContext)
twpr.writeWordCountsToCSV(topics)
# done




















topics_dict = {}
for i in range(num_topics):
    topic_name = '' # topic_name is a string
    topic_name = 'topic'+str(i+1) # a string such as topic1 or topic25
    # create word counts for this topic (index position in all topic distros)
    for distro_row in distros_all:
        tweet_id = distro_row[0] # id
        distro_values = distro_row[1:-1] # distributions, not id
        # get the corresponding tweet text
        query = "SELECT text as tweet_text FROM tweets WHERE id_str = "+str(tweet_id)
        text = sqlContext.sql(query).take(1)[0].tweet_text
        word_array = twpr.normalizeAndSplit(text)
        for word in word_array:
            if topics_dict[topic_name][word] is not None:
                # add the probability for this word
                topics_dict[topic_name][word] = topics_array[topic_name][word]+distro_values[i]
            else:
                # initialize with the probability for this word
                topics_dict =
                topics_dict[topic_name][word] = distro_values[i]
# done



''' The idea here is to do something with TF-IDF frequencies for the word counts'''
tex = tweet_texts.take(1)[0].text
vect = twpr.docToVector(dictionary, tex)

tfidf = gensim.models.tfidfmodel.TfidfModel(corpus) # TF-IDF model from corpus
values = tfidf[vect] # ==> (0, 0.30445337564088687), (1, 0.43182225094537247) ..]




def selectMostProbableTopic(rdd_row, num_topics):
    import numpy as np
    # rdd_object: [(tw.id_str, probVect=[(float), (float), ..]]
    topic_probs = np.array(rdd_row[1:-1])
    max_prob = np.amax(topic_probs)
    if (np.sum((max_prob == topic_probs)) == 1:
        # (max_prob == topic_probs) ==> array([ True, False, False,  True], dtype=bool)
        # np.sum((max_prob == topic_probs)) ==> 2 (amount of max values)
        # so if only one max value, we know what the topic is
        # for these rows, store in DB
        index_prob = np.argmax(topic_probs) # this should return the index of the maximum
        topic_number = topic_probs[index_prob]
        return int(topic_number)
    else :
        # do not store this tweet since no topic was found..
        return False

def storeRDDtoDB(rdd_object, num_topics):
    # rdd_object: [(tw.id_str, probVect=[(float), (float), ..]]
    import psycopg2

    # connect to database
    try:
        conn = psycopg2.connect("dbname='gh_tweets' user='timo' host='localhost' ") # password=''
        cur = conn.cursor() # cursor for psycopg2
    except:
        print("I am unable to connect to the database")

    # execute a query
    # try:
    #     cur.execute("""SELECT * FROM users;""") # query
    #     rows = cur.fetchall()
    # except DatabaseError:
    #     conn.rollback()

    try:
        for u in rdd_object.collect():
            query = "SELECT text as tweet_text, id_str as tw_id_str, user.id_str as tw_id_str, user.name as tw_user_handle FROM tweets WHERE id_str = "+str(u[0])
            tweet = sqlContext.sql(query).take(1)[0]
            print(tweet)
            # next, select most proable topic
            topic_no = selectMostProbableTopic(u, num_topics)
            if not topic_no:
                # do not store this tweet!
            else:
                if topic_no > 0:
                    #num_topics store also!
                    '''----- TODO !!!!! '''
                    cur.execute(
                        """INSERT INTO users (tw_id_str, sp_tweet_count)
                            VALUES (%s, %s);""",
                            (u.user_id, u.tweet_count))
        conn.commit()
        cur.execute("""SELECT * FROM users;""") # query
        rows = cur.fetchall()
        print(rows)
    except AttributeError as e:
        print("Attribute Error: ", e)
        print(u)
        conn.rollback()
    except:
        e = sys.exc_info()[0]
        print("Error: ", e)
        conn.rollback()

# aka distrosToDB
storeRDDtoDB(distros, num_topics)

#dictionary.save('/tmp/tweets.dict')
#corpora.MmCorpus.serialize('/tmp/tweets.mm', corpus)

#dictionary = corpora.Dictionary.load('/tmp/tweets.dict')
#corpus = corpora.MmCorpus('/tmp/tweets.mm')


distro_nums = [25,50]
for i in distro_nums:
    distrosToDB(i)
