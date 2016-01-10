# takes a unicode string (a tweet) and removes stopwords, then returns a string without them
def removeStopWords(tweet, stopword_list):
    tweet_list = tweet.split()
    return " ".join([x for x in tweet_list if x.lower() not in stopword_list])
#done

# make key-value tuples from a text, with values always being 1
# this makes the tweet be ready for combineByKey
def wordsFromText(tweet):
    word_list = tweet.split()
    words = []
    for i in word_list:
        words.append( (i.lower(), 1) )
    return words



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
        raise
    else:
        # yes
        #ascii_tweet = unicodedata.normalize('NFKD', tweet).encode('ascii','ignore')
        if type(tweet) is str:
            ascii_tweet = tweet
        else:
            try:
                ascii_tweet = tweet.decode().rstrip('\n')
            except AttributeError as err:
                print("Attribute Error: "+str(err))
                print(tweet)
                print("Type is: "+str(type(tweet)))
        import re # needed for stripping text of special characters
        # drop special characters
        nospec_tweet = re.sub('[\n\t\r\v\f;\"\']+|[^a-zA-Z\d\s:/]+|[\/.:?=_&#!]+|http+|https+|www+|.com+|bit.ly', '', ascii_tweet)
        # lowercase and split
        norm_tweet = nospec_tweet.lower()
        # if nothing remains, return empty array
        if len(norm_tweet) > 0:
            return norm_tweet.split()
        else:
            return []
# done

def normalizeAndSplitWithStemmingDeprecated(tweet):
    import nltk
    # TODO this attempt to catch faulty input does not work!
    # if type(tweet) is 'pyspark.sql.types.Row':
    #     raise AttributeError('You are trying to give a Spark SQL Row, you need to be more specific!')
    import re # needed for stripping text of special characters
    ascii_tweet = ''
    if type(tweet) is bytes:
        print("This is bytes: ", tweet, " and we need to change it.")
        raise
    else:
        # yes
        #ascii_tweet = unicodedata.normalize('NFKD', tweet).encode('ascii','ignore')
        if type(tweet) is str:
            ascii_tweet = tweet
        else:
            try:
                ascii_tweet = tweet.decode().rstrip('\n')
            except AttributeError as err:
                print("Attribute Error: "+str(err))
                print(tweet)
                print("Type is: "+str(type(tweet)))
        # start stemming and all that
        lcase_tweet = ascii_tweet.lower()
        from ttp import ttp
        # twitter tweet cleaner
        parsed_tweet = ttp.escape(lcase_tweet)
        import re # needed for stripping text of special characters
        nospec_tweet = re.sub('[\n\t\r\v\f;\"\']+|[^a-zA-Z\d\s:/]+|[\/.:?=_&#!]+|http+|https+|www+|.com+|bit.ly', '', parsed_tweet)
        # start stemming
        import nltk.stem.porter as porter
        stemmer = porter.PorterStemmer()
        words = nospec_tweet.split()
        # stemming for each word
        out_array = []
        for w in words:
            stemmed_word = stemmer.stem(w)
            if len(stemmed_word) > 1:
                out_array.append(stemmed_word)
        # done
        return out_array
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
        raise
    else:
        # yes
        #ascii_tweet = unicodedata.normalize('NFKD', tweet).encode('ascii','ignore')
        if type(tweet) is str:
            ascii_tweet = tweet
        else:
            try:
                ascii_tweet = tweet.decode().rstrip('\n')
            except AttributeError as err:
                print("Attribute Error: "+str(err))
                print(tweet)
                print("Type is: "+str(type(tweet)))
        # start stemming and all that
        lcase_tweet = ascii_tweet.lower()
        from ttp import ttp
        # twitter tweet cleaner
        parsed_tweet = ttp.escape(lcase_tweet)
        import re # needed for stripping text of special characters
        nospec_tweet = re.sub('[\n\t\r\v\f;\"\']+|[^a-zA-Z\d\s:/]+|[\/.:?=_&#!]+|http+|https+|www+|.com+|bit.ly', '', parsed_tweet)
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
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, update_every=0, passes=20)
    # Now that the topic model is built (variable named lda),
    # next we want to iterate over each tweet (after normalizing the tweet)
    # and see in which topic that specific tweet goes to, and then to produce tuples of tweet ID and topic ID
    distros = tweet_ids.map(lambda tw: (tw.id, printProbVect(lda, dictionary, tw.text)) )
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

def produce_tfidf_topics(arr, tpcs, dist, corpus, dictionary, tfidf):
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

def wordCountFromTopicDistributions(distros_all, sqlContext):
    topics = {}
    counter = 0
    for tweet in distros_all:
        length = len(distros_all)
        ldadist = tweet[1] # prob distribution
        tweet_id = tweet[0]
        query = "SELECT text as tweet_text FROM tweets WHERE id_str = "+str(tweet_id)
        text = sqlContext.sql(query).take(1)[0].tweet_text
        print('Starting to analyze: '+text)
        stopwords = ["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","a's","accordingly","again","allows","also","amongst","anybody","anyways","appropriate","aside","available","because","before","below","between","by","can't","certain","com","consider","corresponding","definitely","different","don't","each","else","et","everybody","exactly","fifth","follows","four","gets","goes","greetings","has","he","her","herein","him","how","i'm","immediate","indicate","instead","it","itself","know","later","lest","likely","ltd","me","more","must","nd","needs","next","none","nothing","of","okay","ones","others","ourselves","own","placed","probably","rather","regarding","right","saying","seeing","seen","serious","she","so","something","soon","still","t's","th","that","theirs","there","therein","they'd","third","though","thus","toward","try","under","unto","used","value","vs","way","we've","weren't","whence","whereas","whether","who's","why","within","wouldn't","you'll","yourself","able","across","against","almost","although","an","anyhow","anywhere","are","ask","away","become","beforehand","beside","beyond","c'mon","cannot","certainly","come","considering","could","described","do","done","edu","elsewhere","etc","everyone","example","first","for","from","getting","going","had","hasn't","he's","here","hereupon","himself","howbeit","i've","in","indicated","into","it'd","just","known","latter","let","little","mainly","mean","moreover","my","near","neither","nine","noone","novel","off","old","only","otherwise","out","particular","please","provides","rd","regardless","said","says","seem","self","seriously","should","some","sometime","sorry","sub","take","than","that's","them","there's","theres","they'll","this","three","to","towards","trying","unfortunately","up","useful","various","want","we","welcome","what","whenever","whereby","which","whoever","will","without","yes","you're","yourselves","about","actually","ain't","alone","always","and","anyone","apart","aren't","asking","awfully","becomes","behind","besides","both","c's","cant","changes","comes","contain","couldn't","despite","does","down","eg","enough","even","everything","except","five","former","further","given","gone","hadn't","have","hello","here's","hers","his","however","ie","inasmuch","indicates","inward","it'll","keep","knows","latterly","let's","look","many","meanwhile","most","myself","nearly","never","no","nor","now","often","on","onto","ought","outside","particularly","plus","que","re","regards","same","second","seemed","selves","seven","shouldn't","somebody","sometimes","specified","such","taken","thank","thats","themselves","thereafter","thereupon","they're","thorough","through","together","tried","twice","unless","upon","uses","very","wants","we'd","well","what's","where","wherein","while","whole","willing","won't","yet","you've","zero","above","after","all","along","am","another","anything","appear","around","associated","be","becoming","being","best","brief","came","cause","clearly","concerning","containing","course","did","doesn't","downwards","eight","entirely","ever","everywhere","far","followed","formerly","furthermore","gives","got","happens","haven't","help","hereafter","herself","hither","i'd","if","inc","inner","is","it's","keeps","last","least","like","looking","may","merely","mostly","name","necessary","nevertheless","nobody","normally","nowhere","oh","once","or","our","over","per","possible","quite","really","relatively","saw","secondly","seeming","sensible","several","since","somehow","somewhat","specify","sup","tell","thanks","the","then","thereby","these","they've","thoroughly","throughout","too","tries","two","unlikely","us","using","via","was","we'll","went","whatever","where's","whereupon","whither","whom","wish","wonder","you","your","according","afterwards","allow","already","among","any","anyway","appreciate","as","at","became","been","believe","better","but","can","causes","co","consequently","contains","currently","didn't","doing","during","either","especially","every","ex","few","following","forth","get","go","gotten","hardly","having","hence","hereby","hi","hopefully","i'll","ignored","indeed","insofar","isn't","its","kept","lately","less","liked","looks","maybe","might","much","namely","need","new","non","not","obviously","ok","one","other","ours","overall","perhaps","presumably","qv","reasonably","respectively","say","see","seems","sent","shall","six","someone","somewhere","specifying","sure","tends","thanx","their","thence","therefore","they","think","those","thru","took","truly","un","until","use","usually","viz","wasn't","we're","were","when","whereafter","wherever","who","whose","with","would","you'd","yours","I","a","about","an","are","as","at","be","by","com","for","from","how","in","is","it","of","on","or","that","the","this","to","was","what","when","where","who","will","with","the","www","a","able","about","above","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","after","afterwards","again","against","ah","all","almost","alone","along","already","also","although","always","am","among","amongst","an","and","announce","another","any","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","are","aren","arent","arise","around","as","aside","ask","asking","at","auth","available","away","awfully","b","back","be","became","because","become","becomes","becoming","been","before","beforehand","begin","beginning","beginnings","begins","behind","being","believe","below","beside","besides","between","beyond","biol","both","brief","briefly","but","by","c","ca","came","can","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","could","couldnt","d","date","did","didn't","different","do","does","doesn't","doing","done","don't","down","downwards","due","during","e","each","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","et-al","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","few","ff","fifth","first","five","fix","followed","following","follows","for","former","formerly","forth","found","four","from","further","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","had","happens","hardly","has","hasn't","have","haven't","having","he","hed","hence","her","here","hereafter","hereby","herein","heres","hereupon","hers","herself","hes","hi","hid","him","himself","his","hither","home","how","howbeit","however","hundred","i","id","ie","if","i'll","im","immediate","immediately","importance","important","in","inc","indeed","index","information","instead","into","invention","inward","is","isn't","it","itd","it'll","its","itself","i've","j","just","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","m","made","mainly","make","makes","many","may","maybe","me","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","more","moreover","most","mostly","mr","mrs","much","mug","must","my","myself","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","no","nobody","non","none","nonetheless","noone","nor","normally","nos","not","noted","nothing","now","nowhere","o","obtain","obtained","obviously","of","off","often","oh","ok","okay","old","omitted","on","once","one","ones","only","onto","or","ord","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","owing","own","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","re","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","s","said","same","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","she","shed","she'll","shes","should","shouldn't","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","so","some","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","such","sufficiently","suggest","sup","sure"]
        # generate list of top 50 most common words from all tweets and remove them
        #common_words = twpr.genCommonWordsList(tweet_texts, 50)
        # remove stopwords
        text_wosw = removeStopWords(text, stopwords)
        # remove common words
        #text_wocw = twpr.removeCommonWords(text_wosw, common_words)
        # break down for word counts
        word_array = normalizeAndSplitWithLemmatization(text_wosw)
        # stemming?
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
        print('Starting to analyze: '+text)
        stopwords = ["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","a's","accordingly","again","allows","also","amongst","anybody","anyways","appropriate","aside","available","because","before","below","between","by","can't","certain","com","consider","corresponding","definitely","different","don't","each","else","et","everybody","exactly","fifth","follows","four","gets","goes","greetings","has","he","her","herein","him","how","i'm","immediate","indicate","instead","it","itself","know","later","lest","likely","ltd","me","more","must","nd","needs","next","none","nothing","of","okay","ones","others","ourselves","own","placed","probably","rather","regarding","right","saying","seeing","seen","serious","she","so","something","soon","still","t's","th","that","theirs","there","therein","they'd","third","though","thus","toward","try","under","unto","used","value","vs","way","we've","weren't","whence","whereas","whether","who's","why","within","wouldn't","you'll","yourself","able","across","against","almost","although","an","anyhow","anywhere","are","ask","away","become","beforehand","beside","beyond","c'mon","cannot","certainly","come","considering","could","described","do","done","edu","elsewhere","etc","everyone","example","first","for","from","getting","going","had","hasn't","he's","here","hereupon","himself","howbeit","i've","in","indicated","into","it'd","just","known","latter","let","little","mainly","mean","moreover","my","near","neither","nine","noone","novel","off","old","only","otherwise","out","particular","please","provides","rd","regardless","said","says","seem","self","seriously","should","some","sometime","sorry","sub","take","than","that's","them","there's","theres","they'll","this","three","to","towards","trying","unfortunately","up","useful","various","want","we","welcome","what","whenever","whereby","which","whoever","will","without","yes","you're","yourselves","about","actually","ain't","alone","always","and","anyone","apart","aren't","asking","awfully","becomes","behind","besides","both","c's","cant","changes","comes","contain","couldn't","despite","does","down","eg","enough","even","everything","except","five","former","further","given","gone","hadn't","have","hello","here's","hers","his","however","ie","inasmuch","indicates","inward","it'll","keep","knows","latterly","let's","look","many","meanwhile","most","myself","nearly","never","no","nor","now","often","on","onto","ought","outside","particularly","plus","que","re","regards","same","second","seemed","selves","seven","shouldn't","somebody","sometimes","specified","such","taken","thank","thats","themselves","thereafter","thereupon","they're","thorough","through","together","tried","twice","unless","upon","uses","very","wants","we'd","well","what's","where","wherein","while","whole","willing","won't","yet","you've","zero","above","after","all","along","am","another","anything","appear","around","associated","be","becoming","being","best","brief","came","cause","clearly","concerning","containing","course","did","doesn't","downwards","eight","entirely","ever","everywhere","far","followed","formerly","furthermore","gives","got","happens","haven't","help","hereafter","herself","hither","i'd","if","inc","inner","is","it's","keeps","last","least","like","looking","may","merely","mostly","name","necessary","nevertheless","nobody","normally","nowhere","oh","once","or","our","over","per","possible","quite","really","relatively","saw","secondly","seeming","sensible","several","since","somehow","somewhat","specify","sup","tell","thanks","the","then","thereby","these","they've","thoroughly","throughout","too","tries","two","unlikely","us","using","via","was","we'll","went","whatever","where's","whereupon","whither","whom","wish","wonder","you","your","according","afterwards","allow","already","among","any","anyway","appreciate","as","at","became","been","believe","better","but","can","causes","co","consequently","contains","currently","didn't","doing","during","either","especially","every","ex","few","following","forth","get","go","gotten","hardly","having","hence","hereby","hi","hopefully","i'll","ignored","indeed","insofar","isn't","its","kept","lately","less","liked","looks","maybe","might","much","namely","need","new","non","not","obviously","ok","one","other","ours","overall","perhaps","presumably","qv","reasonably","respectively","say","see","seems","sent","shall","six","someone","somewhere","specifying","sure","tends","thanx","their","thence","therefore","they","think","those","thru","took","truly","un","until","use","usually","viz","wasn't","we're","were","when","whereafter","wherever","who","whose","with","would","you'd","yours","I","a","about","an","are","as","at","be","by","com","for","from","how","in","is","it","of","on","or","that","the","this","to","was","what","when","where","who","will","with","the","www","a","able","about","above","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","after","afterwards","again","against","ah","all","almost","alone","along","already","also","although","always","am","among","amongst","an","and","announce","another","any","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","are","aren","arent","arise","around","as","aside","ask","asking","at","auth","available","away","awfully","b","back","be","became","because","become","becomes","becoming","been","before","beforehand","begin","beginning","beginnings","begins","behind","being","believe","below","beside","besides","between","beyond","biol","both","brief","briefly","but","by","c","ca","came","can","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","could","couldnt","d","date","did","didn't","different","do","does","doesn't","doing","done","don't","down","downwards","due","during","e","each","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","et-al","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","few","ff","fifth","first","five","fix","followed","following","follows","for","former","formerly","forth","found","four","from","further","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","had","happens","hardly","has","hasn't","have","haven't","having","he","hed","hence","her","here","hereafter","hereby","herein","heres","hereupon","hers","herself","hes","hi","hid","him","himself","his","hither","home","how","howbeit","however","hundred","i","id","ie","if","i'll","im","immediate","immediately","importance","important","in","inc","indeed","index","information","instead","into","invention","inward","is","isn't","it","itd","it'll","its","itself","i've","j","just","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","m","made","mainly","make","makes","many","may","maybe","me","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","more","moreover","most","mostly","mr","mrs","much","mug","must","my","myself","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","no","nobody","non","none","nonetheless","noone","nor","normally","nos","not","noted","nothing","now","nowhere","o","obtain","obtained","obviously","of","off","often","oh","ok","okay","old","omitted","on","once","one","ones","only","onto","or","ord","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","owing","own","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","re","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","s","said","same","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","she","shed","she'll","shes","should","shouldn't","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","so","some","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","such","sufficiently","suggest","sup","sure"]
        # generate list of top 50 most common words from all tweets and remove them
        #common_words = twpr.genCommonWordsList(tweet_texts, 50)
        # remove stopwords
        text_wosw = removeStopWords(text, stopwords)
        # remove common words
        #text_wocw = twpr.removeCommonWords(text_wosw, common_words)
        # break down for word counts
        word_array = normalizeAndSplitWithLemmatization(text_wosw)
        # stemming?
        topics = produce_tfidf_topics(word_array, topics, ldadist, corpus, dictionary, tfidf_model)
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
    stopwords = ["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","a's","accordingly","again","allows","also","amongst","anybody","anyways","appropriate","aside","available","because","before","below","between","by","can't","certain","com","consider","corresponding","definitely","different","don't","each","else","et","everybody","exactly","fifth","follows","four","gets","goes","greetings","has","he","her","herein","him","how","i'm","immediate","indicate","instead","it","itself","know","later","lest","likely","ltd","me","more","must","nd","needs","next","none","nothing","of","okay","ones","others","ourselves","own","placed","probably","rather","regarding","right","saying","seeing","seen","serious","she","so","something","soon","still","t's","th","that","theirs","there","therein","they'd","third","though","thus","toward","try","under","unto","used","value","vs","way","we've","weren't","whence","whereas","whether","who's","why","within","wouldn't","you'll","yourself","able","across","against","almost","although","an","anyhow","anywhere","are","ask","away","become","beforehand","beside","beyond","c'mon","cannot","certainly","come","considering","could","described","do","done","edu","elsewhere","etc","everyone","example","first","for","from","getting","going","had","hasn't","he's","here","hereupon","himself","howbeit","i've","in","indicated","into","it'd","just","known","latter","let","little","mainly","mean","moreover","my","near","neither","nine","noone","novel","off","old","only","otherwise","out","particular","please","provides","rd","regardless","said","says","seem","self","seriously","should","some","sometime","sorry","sub","take","than","that's","them","there's","theres","they'll","this","three","to","towards","trying","unfortunately","up","useful","various","want","we","welcome","what","whenever","whereby","which","whoever","will","without","yes","you're","yourselves","about","actually","ain't","alone","always","and","anyone","apart","aren't","asking","awfully","becomes","behind","besides","both","c's","cant","changes","comes","contain","couldn't","despite","does","down","eg","enough","even","everything","except","five","former","further","given","gone","hadn't","have","hello","here's","hers","his","however","ie","inasmuch","indicates","inward","it'll","keep","knows","latterly","let's","look","many","meanwhile","most","myself","nearly","never","no","nor","now","often","on","onto","ought","outside","particularly","plus","que","re","regards","same","second","seemed","selves","seven","shouldn't","somebody","sometimes","specified","such","taken","thank","thats","themselves","thereafter","thereupon","they're","thorough","through","together","tried","twice","unless","upon","uses","very","wants","we'd","well","what's","where","wherein","while","whole","willing","won't","yet","you've","zero","above","after","all","along","am","another","anything","appear","around","associated","be","becoming","being","best","brief","came","cause","clearly","concerning","containing","course","did","doesn't","downwards","eight","entirely","ever","everywhere","far","followed","formerly","furthermore","gives","got","happens","haven't","help","hereafter","herself","hither","i'd","if","inc","inner","is","it's","keeps","last","least","like","looking","may","merely","mostly","name","necessary","nevertheless","nobody","normally","nowhere","oh","once","or","our","over","per","possible","quite","really","relatively","saw","secondly","seeming","sensible","several","since","somehow","somewhat","specify","sup","tell","thanks","the","then","thereby","these","they've","thoroughly","throughout","too","tries","two","unlikely","us","using","via","was","we'll","went","whatever","where's","whereupon","whither","whom","wish","wonder","you","your","according","afterwards","allow","already","among","any","anyway","appreciate","as","at","became","been","believe","better","but","can","causes","co","consequently","contains","currently","didn't","doing","during","either","especially","every","ex","few","following","forth","get","go","gotten","hardly","having","hence","hereby","hi","hopefully","i'll","ignored","indeed","insofar","isn't","its","kept","lately","less","liked","looks","maybe","might","much","namely","need","new","non","not","obviously","ok","one","other","ours","overall","perhaps","presumably","qv","reasonably","respectively","say","see","seems","sent","shall","six","someone","somewhere","specifying","sure","tends","thanx","their","thence","therefore","they","think","those","thru","took","truly","un","until","use","usually","viz","wasn't","we're","were","when","whereafter","wherever","who","whose","with","would","you'd","yours","I","a","about","an","are","as","at","be","by","com","for","from","how","in","is","it","of","on","or","that","the","this","to","was","what","when","where","who","will","with","the","www","a","able","about","above","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","after","afterwards","again","against","ah","all","almost","alone","along","already","also","although","always","am","among","amongst","an","and","announce","another","any","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","are","aren","arent","arise","around","as","aside","ask","asking","at","auth","available","away","awfully","b","back","be","became","because","become","becomes","becoming","been","before","beforehand","begin","beginning","beginnings","begins","behind","being","believe","below","beside","besides","between","beyond","biol","both","brief","briefly","but","by","c","ca","came","can","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","could","couldnt","d","date","did","didn't","different","do","does","doesn't","doing","done","don't","down","downwards","due","during","e","each","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","et-al","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","few","ff","fifth","first","five","fix","followed","following","follows","for","former","formerly","forth","found","four","from","further","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","had","happens","hardly","has","hasn't","have","haven't","having","he","hed","hence","her","here","hereafter","hereby","herein","heres","hereupon","hers","herself","hes","hi","hid","him","himself","his","hither","home","how","howbeit","however","hundred","i","id","ie","if","i'll","im","immediate","immediately","importance","important","in","inc","indeed","index","information","instead","into","invention","inward","is","isn't","it","itd","it'll","its","itself","i've","j","just","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","m","made","mainly","make","makes","many","may","maybe","me","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","more","moreover","most","mostly","mr","mrs","much","mug","must","my","myself","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","no","nobody","non","none","nonetheless","noone","nor","normally","nos","not","noted","nothing","now","nowhere","o","obtain","obtained","obviously","of","off","often","oh","ok","okay","old","omitted","on","once","one","ones","only","onto","or","ord","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","owing","own","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","re","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","s","said","same","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","she","shed","she'll","shes","should","shouldn't","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","so","some","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","such","sufficiently","suggest","sup","sure"]
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
