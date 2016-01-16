# since the stopword list is 1000+ words long,
# it's best to not store it in the script file but to load it
# with pickle from a pickle-encoded python list
def loadStopWordsFromFile(filename='stopwords.pkl'):
    import pickle
    pkl_file = open(filename, 'rb')
    stopwords = pickle.load(pkl_file)
    return stopwords

# takes a unicode string (a tweet) and removes stopwords, then returns a string without them
def removeStopWords(tweet, stopword_list):
    tweet_list = tweet.split()
    return " ".join([x for x in tweet_list if x.lower() not in stopword_list])
#done

def unicodeToASCII(text=''):
    import re
    out = text.decode().rstrip('\n')
    return out

# make key-value tuples from a text, with values always being 1
# this makes the tweet be ready for combineByKey
def wordsFromText(tweet):
    word_list = tweet.split()
    words = []
    for i in word_list:
        key_value_tuple = (i.lower(), 1)
        words.append(key_value_tuple)
    return words

def removeSpecialCharactersAndSomeURLs(text=''):
    import re
    out = re.sub('[\n\t\r\v\f;\"\']+|[^a-zA-Z\d\s:/]+|[/.:?=_&#!]+|http+|https+|www+|.com+|bit.ly', '', text)
    return out

# THIS COULD BE DEPRECATED AND USE THE gensim.dictionary.filter_extremes function instead!
def removeRareWords(tweet, rare_words_list):
    tweet_list = tweet.split()
    return " ".join([x for x in tweet_list if x.lower() not in rare_words_list])
#done

def genCommonWordsList(tweets, amount):
    common_words_list = []
    tw_sorted = sorted(tweets, key=lambda x: x[1])
    cursor = 0
    while cursor <= amount:
        common_words_list.append(tw_sorted[-cursor][0])
        cursor = cursor + 1
    return common_words_list
#done

# THIS COULD BE DEPRECATED AND USE THE gensim.dictionary.filter_extremes function instead!
def removeCommonWords(tweet, common_words_list):
    tweet_list = tweet.split()
    return " ".join([x for x in tweet_list if x.lower() not in common_words_list])
#done

# now we have the tw_woswarw (tweets without stopwords and rare words) list of tweets as lists of strings ready to be processed by gensim
# also, remove special characters

def normalizeAndSplit(tweet):
    # TODO this attempt to catch faulty input does not work!
    # if type(tweet) is 'pyspark.sql.types.Row':
    #     raise AttributeError('You are trying to give a Spark SQL Row, you need to be more specific!')
    import re # needed for stripping text of special characters
    ascii_tweet = ''
    if type(tweet) is bytes:
        print("This is bytes: ", tweet, " and we need to change it.")
        raise Exception("Tweet has type of bytes. Should have type of string.")
    else:
        # yes
        #ascii_tweet = unicodedata.normalize('NFKD', tweet).encode('ascii','ignore')
        if type(tweet) is str:
            ascii_tweet = tweet
        else:
            try:
                ascii_tweet = unicodeToASCII(tweet)
            except AttributeError as err:
                print("Attribute Error: "+str(err))
                print(tweet)
                print("Type is: "+str(type(tweet)))
        # drop special characters
        nospec_tweet = removeSpecialCharactersAndSomeURLs(ascii_tweet)
        # lowercase and split
        norm_tweet = nospec_tweet.lower()
        # if nothing remains, return empty array
        if len(norm_tweet) > 0:
            return norm_tweet.split()
        else:
            return []
# done

def normalizeAndSplitWithLemmatization(tweet):
    import nltk
    # TODO this attempt to catch faulty input does not work!
    # if type(tweet) is 'pyspark.sql.types.Row':
    #     raise AttributeError('You are trying to give a Spark SQL Row, you need to be more specific!')
    import re # needed for stripping text of special characters
    ascii_tweet = ''
    if type(tweet) is bytes:
        print("This is bytes: ", tweet, " and we need to change it.")
        raise Exception("Tweet has type of bytes. Should have type of string.")
    else:
        # yes
        #ascii_tweet = unicodedata.normalize('NFKD', tweet).encode('ascii','ignore')
        if type(tweet) is str:
            ascii_tweet = tweet
        else:
            try:
                ascii_tweet = unicodeToASCII(tweet)
            except AttributeError as err:
                print("Attribute Error: "+str(err))
                print(tweet)
                print("Type is: "+str(type(tweet)))
        # start stemming and all that
        lcase_tweet = ascii_tweet.lower()
        from ttp import ttp
        # twitter tweet cleaner
        parsed_tweet = ttp.escape(lcase_tweet)
        nospec_tweet = removeSpecialCharactersAndSomeURLs(parsed_tweet)
        # start stemming
        from nltk.stem import WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()
        words = nospec_tweet.split()
        # lemmatization for each word
        out_array = []
        for w in words:
            lemmatized_word = wordnet_lemmatizer.lemmatize(w)
            if len(lemmatized_word) > 1:
                out_array.append(lemmatized_word)
        # done
        return out_array
# done

def docToVector(dictionary, doc):
    return dictionary.doc2bow(normalizeAndSplit(doc))

def printProbVect(ldaModel, dictionary, vec):
    return ldaModel[docToVector(dictionary, vec)]

def doLDA(corpus, dictionary, num_topics, tweet_ids):
    import re # needed for stripping text of special characters
    import unicodedata # needed for unicode to ASCII conversion without errors
    import gensim # needed for text clustering
    from gensim import corpora, models, similarities # needed for text clustering
    import sklearn
    # extract LDA topics
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, update_every=0, passes=20)
    # Now that the topic model is built (variable named lda),
    # next we want to iterate over each tweet (after normalizing the tweet)
    # and see in which topic that specific tweet goes to, and then to produce tuples of tweet ID and topic ID
    distros = tweet_ids.map(lambda tw: (tw.id, printProbVect(lda, dictionary, tw.text)))
    return distros

def produce_topics(arr, tpcs, dist):
    word_array = arr # array of words in a given tweet to be counted to topics
    ldadist = dist # distributions of probabilities for each topic, used for incrementing the output counts
    topics = tpcs # output object, dictionary of topics
    # target: {topic1: {growthhacking: 100.2, startups: 34.2}, topic2: {contentmarketing: 43.4, other: 32.3} }
    # start iterating over words in input word array
    for word in word_array:
        # for each word, see probabilitiy of belonging to each topic
        for i in ldadist:
            # [(1,0.83), (2, 0.38), (23, 0.11)] <-- probabilities of a word belonging to a topic
            i_topic = i[0] # zero-index number of topic, e.g. 0..24 for a 25-topic model
            i_value = float(i[1]) # probability, a float, e.g. 0.234
            topic_name = "topic"+str(i_topic+1)
            # target: {topic1: {growthhacking: 100.2, startups: 34.2}, topic2: {contentmarketing: 43.4, other: 32.3} }
            # 1. find or create topic
            if topic_name not in list(topics.keys()): # if topic already exists
                topics[topic_name] = {}
            # 2. find or create word and attached value
            if word in list(topics[topic_name].keys()): # word we're about to add is not included, so need to add
                # word exists, increment
                existing_value = topics[topic_name][word]
                topics[topic_name][word] = existing_value+i_value
            else: # word is not found so need to create key-value pair with fresh value
                topics[topic_name][word] = i_value
    return topics

def produce_tfidf_topics(arr, tpcs, dist, dictionary, tfidf):
    word_array = arr # array of words in a given tweet to be counted to topics
    tfidfs = [] # array of tuples (id, tfidf, dictionary_word_id) that are in the dictionary
    ldadist = dist # distributions of probabilities for each topic, used for incrementing the output counts
    topics = tpcs # output object, dictionary of topics
    # target: {topic1: {growthhacking: 100.2, startups: 34.2}, topic2: {contentmarketing: 43.4, other: 32.3} }
    # start iterating over words in input word array
    print("\n\n -- Word array:")
    print(word_array)
    docb = dictionary.doc2bow(word_array)
    print("TFIDFS:")
    tfs = tfidf[docb]
    print(tfs)
    for tfer in tfs:
        id_key = tfer[0]
        val_value = tfer[1]
        tf_word = dictionary.get(id_key)
        tfidfs.append( (tf_word, val_value, id_key) )
    print("OUTPUT:")
    print(tfidfs)
    print("Starting iteration:")
    # values initialized
    for word in tfidfs:
        # for each word, see probabilitiy of belonging to each topic
        for i in ldadist:
            # [(1,0.83), (2, 0.38), (23, 0.11)] <-- probabilities of a tweet belonging to a topic
            word_name = word[0]
            i_topic = i[0] # zero-index number of topic, e.g. 0..24 for a 25-topic model
            i_value = float(word[1]) # tf-idf value, a float, e.g. 0.234
            topic_name = "topic"+str(i_topic+1)
            print("final output:", str(i_topic), word_name, str(i_value) )
            # target: {topic1: {growthhacking: 100.2, startups: 34.2}, topic2: {contentmarketing: 43.4, other: 32.3} }
            # 1. find or create topic
            if topic_name not in list(topics.keys()): # if topic already exists
                topics[topic_name] = {}
            # 2. find or create word and attached value
            if word_name in list(topics[topic_name].keys()): # word we're about to add is not included, so need to add
                # word exists, increment
                existing_value = topics[topic_name][word_name]
                topics[topic_name][word_name] = existing_value+i_value
            else: # word is not found so need to create key-value pair with fresh value
                topics[topic_name][word_name] = i_value
    return topics

def preprocessStopWordsSplitLemmatize(text):
    stopwords = loadStopWordsFromFile()
    text_wosw = removeStopWords(text, stopwords)
    word_array = normalizeAndSplitWithLemmatization(text_wosw)
    return word_array

def wordCountFromTopicDistributions(distros_all, sqlContext):
    topics = {}
    counter = 0
    for tweet in distros_all:
        length = len(distros_all)
        ldadist = tweet[1] # prob distribution
        tweet_id = tweet[0]
        query = "SELECT text as tweet_text FROM tweets WHERE id_str = "+str(tweet_id)
        text = sqlContext.sql(query).take(1)[0].tweet_text
        word_array = preprocessStopWordsSplitLemmatize(text)
        topics = produce_topics(word_array, topics, ldadist)
        counter = counter+1
        print(" -- Analysis for tweet "+str(counter)+"/"+str(length)+" completed.")
    return topics

def TFIDFsFromTopicDistributions(distros_all, sqlContext, corpus, dictionary):
    import gensim # needed for text clustering
    from gensim import corpora, models, similarities # needed for text clustering
    tfidf_model = models.TfidfModel(corpus)
    topics = {}
    counter = 0
    for tweet in distros_all:
        length = len(distros_all)
        ldadist = tweet[1] # prob distribution
        tweet_id = tweet[0]
        query = "SELECT text as tweet_text FROM tweets WHERE id_str = "+str(tweet_id)
        text = sqlContext.sql(query).take(1)[0].tweet_text
        word_array = preprocessStopWordsSplitLemmatize(text)
        topics = produce_tfidf_topics(word_array, topics, ldadist, dictionary, tfidf_model)
        counter = counter+1
        print(" -- Analysis for tweet "+str(counter)+"/"+str(length)+" completed.")
    return topics

def writeTFIDFsToCSV(topics):
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

def writeWordCountsToCSV(topics):
    import csv
    for topic in topics:
        fname = str(topic)+'.csv'
        with open(fname, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            header = ['word', 'count']
            writer.writerow(header)
            for key in list( topics[topic].keys() ):
                value = topics[topic][key]
                writer.writerow([key, value])

def run(tweet_texts):
    import re
    import unicodedata # needed for unicode to ASCII conversion without errors
    import gensim # needed for text clustering
    from gensim import corpora, models, similarities # needed for text clustering
    import sklearn
    # load stopwords into memory from file
    stopwords = loadStopWordsFromFile()
    # tweets without rare rowrds
    tw_wosws = tweet_texts.map(lambda tw: removeStopWords(tw[0], stopwords))
    # word count for each of the words in the corpus
    # word_count = tw_wosws.flatMap(lambda tweet: wordsFromText(tweet)).reduceByKey(lambda a, b: a+b).collect()
    # create a list out of the rare words in the list
    # rare_words = [x[0] for x in word_count if x[1] <= 1]
    # find common words
    # common_words = genCommonWordsList(word_count, 50)
    # tw_woswarw = tw_wosws.map(lambda tweet: removeRareWords(tweet, rare_words)).map(lambda tweet: removeCommonWords(tweet, common_words)).map(lambda tweet: normalizeAndSplit(tweet) )
    out = tw_wosws.map(lambda tweet: normalizeAndSplitWithLemmatization(tweet) )
    texts = out.collect()
    return texts

def buildDictionaryFromTexts(texts):
    from gensim import corpora
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.3, keep_n=None) # used instead of manually doing this
    return dictionary

def buildCorpusFromDictionaryAndTexts(texts, dictionary):
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus

def doHDP(corpus, dictionary):
    from gensim import models
    hdp = models.hdpmodel.HdpModel(corpus, dictionary)
    # hdp.print_topics(topics=-1, topn=1)
    return hdp

def createTopicWordCounts(num_topics, distros_all, tweet_rdd, sqlContext):
    tweet_rdd.registerTempTable("tweets_for_distros")
    topics_dict = {}
    for i in range(num_topics):
        # cycling through each topic
        topic_name = '' # topic_name is a string
        topic_name = 'topic'+str(i+1) # a string such as topic1 or topic25
        # create word counts for this topic (index position in all topic distros)
        print('\n\n #-- Starting iteration for topic ', topic_name)
        for distro_row in distros_all:
            # debug: print("\n -#--- Starting iteration for new tweet:")
            tweet_id = distro_row[0] # id
            # debug: print("The ID of this tweet is: ",tweet_id)
            distro_values = distro_row[1] # distributions, not id
            # get the corresponding tweet text
            query = "SELECT text as tweet_text FROM tweets_for_distros WHERE id_str = "+str(tweet_id)
            text = sqlContext.sql(query).take(1)[0].tweet_text
            # debug: print("The tweet text is: ", text)
            word_array = normalizeAndSplitWithLemmatization(text)
            matching_topics = [item for item in distro_values if item[0] == i]
            # => [(20, 0.66941098782833874)]
            if len(matching_topics) > 0:
                prob = matching_topics[0][1] # from [(20, 0.669..)] to 0.669..
                # only if the row contains matches with the topic being iterated over
                for word in word_array:
                    try:
                        if topics_dict[topic_name] is None:
                            # debug: print('Attempting to create new topic:', topic_name)
                            topics_dict[topic_name] = {word: prob}
                        else:
                            try:
                                if topics_dict[topic_name][word] is not None:
                                    # add the probability for this word
                                    original_prob = topics_dict[topic_name][word]
                                    topics_dict[topic_name][word] = topics_dict[topic_name][word]+prob
                                    # debug: print("Adding to existing word ", word, "in topic ", topic_name, "the total prob of ", str(topics_dict[topic_name][word]+prob), "from original prob ", str(original_prob), ".")
                                else:
                                    # initialize with the probability for this word
                                    # matching_topic = [item for item in distro_values if item[0] == i]
                                    # debug: print("Creating new word", word, "in topic ", topic_name, ", with prob: ", str(prob))
                                    topics_dict[topic_name][word] = prob
                            except:
                                # debug: print("Creating new word", word, "in topic ", topic_name, ", with prob: ", str(prob))
                                topics_dict[topic_name][word] = prob
                    except:
                        # debug: print('Attempting to create new topic:', topic_name)
                        topics_dict[topic_name] = {word: prob}
    return topics_dict
# done


def sayHello():
    print("This works")
