'''You might need to use python 3 with pyspark to get gensim, numpy and sklearn properly installed on OS X'''

import processing as twpr
from imp import reload as reload # to use: twpr = reload(twpr)
import re # needed for stripping text of special characters
import unicodedata # needed for unicode to ASCII conversion without errors
import gensim # needed for text clustering
from gensim import corpora, models, similarities # needed for text clustering
import sklearn

# for testing: path = "/Users/timo/Ruby/GetTweets/stored_tweets/2015-05-08.json"
#path = "/Users/timo/Ruby/GetTweets/stored_tweets/*"
path = "/Users/timo/Code/GetTweets/stored_tweets/2015-05-08.json"
tweets = sqlContext.read.json(path)
tweets.registerTempTable("tweets")
# tweets without stopwords
tweet_texts = sqlContext.sql("SELECT text FROM tweets")

texts = twpr.run(tweet_texts)

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

tweet_ids = sqlContext.sql("SELECT id_str as id, text FROM tweets")

num_topics = 25

distros = twpr.doLDA(corpus, dictionary, num_topics, tweet_ids)

distros_all = distros.collect()
#done


# for each topic create a dictionary of words with float count of words
#  (topic1, [word1: 32.12, word2: 322.34, ..], topic2: [...])


# for each tweet
    # for each word
        # assign probabilities per topic
            # topics can be arrays within arrays

topics = {}
counter = 0
for tweet in distros_all[1:100]:
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
    text_wosw = twpr.removeStopWords(text, stopwords)
    # remove common words
    #text_wocw = twpr.removeCommonWords(text_wosw, common_words)
    # break down for word counts
    word_array = twpr.normalizeAndSplit(text_wosw)
    # stemming?
    for word in word_array:
        for i in ldadist:
            # [(1,0.83), (2, 0.38), (23, 0.11)] <-- probabilities
            i_topic = i[0] # word
            i_value = float(i[1]) # probability
            # target: {topic1: {growthhacking: 100.2, startups: 34.2}, topic2: {contentmarketing: 43.4, other: 32.3} }

            vals = list(topics['topic'+str(i_topic+1)].values())
            print("Printing values:")
            print(vals)

            if i_topic in vals: # word is already found
                existing_value = topics['topic'+str(i_topic+1)][word]
                topics['topic'+str(i_topic+1)][word] = float(existing_value)+float(i_value)
            else: # word not found
                topics['topic'+str(i_topic+1)] = {word: i_value}
            try:
                topics['topic'+str(i_topic+1)][word] = topics['topic'+str(i_topic+1)][word]+i_value
            except:
                try:
                    topics['topic'+str(i_topic+1)] = {word: i_value}
                except:
                    print(topics['topic'+str(i_topic+1)])
    counter = counter+1
    print(" -- Analysis for tweet "+str(counter)+"/"+str(length)+" completed.")
# done

# write to CSV
twpr.writeWordCountsToCSV(topics)




















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
