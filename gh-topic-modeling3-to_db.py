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

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=5, no_above=0.3, keep_n=None) # used instead of manually doing this
corpus = [dictionary.doc2bow(text) for text in texts]
num_topics = 25
# done

tweet_ids = sqlContext.sql("SELECT id_str as id, text FROM tweets")
#done
distros = twpr.doLDA(corpus, dictionary, num_topics, tweet_ids)
distros_all = distros.collect()
#done

topics = twpr.TFIDFsFromTopicDistributions(distros_all[0:-1], sqlContext, corpus, dictionary)


import csv
with open('tfidf_topics.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    header = ['word', 'tfidf', 'topic']
    for topic in topics:
        topicname = str(topic)
        writer.writerow(header)
        for key in list( topics[topic].keys() ):
            value = topics[topic][key]
            writer.writerow([key, value, topicname])


twpr.writeWordCountsToCSV(topics)

hdp = gensim.models.hdpmodel.HdpModel(corpus, dictionary)
hdp.print_topics(topics=-1, topn=1)

# above this is common to both using LDA directly from Gensim and running the proprietary functions below
# lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, update_every=0, passes=20)



#  gensim.models.tfidfmodel.TfidfModel(corpus=None, id2word=None, dictionary=None, wlocal=<function identity>, wglobal=<function df2idf>, normalize=True)
# tfidf = gensim.models.tfidfmodel.TfidfModel(corpus=corpus, dictionary=dictionary)
# print(tfidf[dictionary.doc2bow(['growth', 'hacking', 'is', 'not', 'the', 'same', 'as', 'a/b', 'testing'])])
# => [(42, 0.22787410177633208), (306, 0.5049139570055937), (380, 0.5735775649749842), (490, 0.6034435074784973)]

# dictionary.get(5)
# => 'indiedev'




#topics_dict = twpr.createTopicWordCounts(num_topics, distros_all[0:150], tweets, sqlContext)

# import numpy
# Apache PySpark RDD.takeSample requires numpy
# distros.takeSample(False, 100) # distros.collect()
# done


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
