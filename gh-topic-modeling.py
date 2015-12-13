from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

'''You might need to use python 3 with pyspark to get gensim, numpy and sklearn properly installed on OS X'''

import re # needed for stripping text of special characters
import unicodedata # needed for unicode to ASCII conversion without errors
import gensim # needed for text clustering
from gensim import corpora, models, similarities # needed for text clustering
import sklearn
import numpy


# for testing: path = "/Users/timo/Ruby/GetTweets/stored_tweets/2015-05-08.json"
path = "/Users/timo/Code/GetTweets/stored_tweets/*"
#path = "/Users/timo/Ruby/GetTweets/stored_tweets/*"
tweets = sqlContext.read.json(path)
tweets.registerTempTable("tweets")


#list of stopwords
stopwords = ["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","a's","accordingly","again","allows","also","amongst","anybody","anyways","appropriate","aside","available","because","before","below","between","by","can't","certain","com","consider","corresponding","definitely","different","don't","each","else","et","everybody","exactly","fifth","follows","four","gets","goes","greetings","has","he","her","herein","him","how","i'm","immediate","indicate","instead","it","itself","know","later","lest","likely","ltd","me","more","must","nd","needs","next","none","nothing","of","okay","ones","others","ourselves","own","placed","probably","rather","regarding","right","saying","seeing","seen","serious","she","so","something","soon","still","t's","th","that","theirs","there","therein","they'd","third","though","thus","toward","try","under","unto","used","value","vs","way","we've","weren't","whence","whereas","whether","who's","why","within","wouldn't","you'll","yourself","able","across","against","almost","although","an","anyhow","anywhere","are","ask","away","become","beforehand","beside","beyond","c'mon","cannot","certainly","come","considering","could","described","do","done","edu","elsewhere","etc","everyone","example","first","for","from","getting","going","had","hasn't","he's","here","hereupon","himself","howbeit","i've","in","indicated","into","it'd","just","known","latter","let","little","mainly","mean","moreover","my","near","neither","nine","noone","novel","off","old","only","otherwise","out","particular","please","provides","rd","regardless","said","says","seem","self","seriously","should","some","sometime","sorry","sub","take","than","that's","them","there's","theres","they'll","this","three","to","towards","trying","unfortunately","up","useful","various","want","we","welcome","what","whenever","whereby","which","whoever","will","without","yes","you're","yourselves","about","actually","ain't","alone","always","and","anyone","apart","aren't","asking","awfully","becomes","behind","besides","both","c's","cant","changes","comes","contain","couldn't","despite","does","down","eg","enough","even","everything","except","five","former","further","given","gone","hadn't","have","hello","here's","hers","his","however","ie","inasmuch","indicates","inward","it'll","keep","knows","latterly","let's","look","many","meanwhile","most","myself","nearly","never","no","nor","now","often","on","onto","ought","outside","particularly","plus","que","re","regards","same","second","seemed","selves","seven","shouldn't","somebody","sometimes","specified","such","taken","thank","thats","themselves","thereafter","thereupon","they're","thorough","through","together","tried","twice","unless","upon","uses","very","wants","we'd","well","what's","where","wherein","while","whole","willing","won't","yet","you've","zero","above","after","all","along","am","another","anything","appear","around","associated","be","becoming","being","best","brief","came","cause","clearly","concerning","containing","course","did","doesn't","downwards","eight","entirely","ever","everywhere","far","followed","formerly","furthermore","gives","got","happens","haven't","help","hereafter","herself","hither","i'd","if","inc","inner","is","it's","keeps","last","least","like","looking","may","merely","mostly","name","necessary","nevertheless","nobody","normally","nowhere","oh","once","or","our","over","per","possible","quite","really","relatively","saw","secondly","seeming","sensible","several","since","somehow","somewhat","specify","sup","tell","thanks","the","then","thereby","these","they've","thoroughly","throughout","too","tries","two","unlikely","us","using","via","was","we'll","went","whatever","where's","whereupon","whither","whom","wish","wonder","you","your","according","afterwards","allow","already","among","any","anyway","appreciate","as","at","became","been","believe","better","but","can","causes","co","consequently","contains","currently","didn't","doing","during","either","especially","every","ex","few","following","forth","get","go","gotten","hardly","having","hence","hereby","hi","hopefully","i'll","ignored","indeed","insofar","isn't","its","kept","lately","less","liked","looks","maybe","might","much","namely","need","new","non","not","obviously","ok","one","other","ours","overall","perhaps","presumably","qv","reasonably","respectively","say","see","seems","sent","shall","six","someone","somewhere","specifying","sure","tends","thanx","their","thence","therefore","they","think","those","thru","took","truly","un","until","use","usually","viz","wasn't","we're","were","when","whereafter","wherever","who","whose","with","would","you'd","yours","I","a","about","an","are","as","at","be","by","com","for","from","how","in","is","it","of","on","or","that","the","this","to","was","what","when","where","who","will","with","the","www","a","able","about","above","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","after","afterwards","again","against","ah","all","almost","alone","along","already","also","although","always","am","among","amongst","an","and","announce","another","any","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","are","aren","arent","arise","around","as","aside","ask","asking","at","auth","available","away","awfully","b","back","be","became","because","become","becomes","becoming","been","before","beforehand","begin","beginning","beginnings","begins","behind","being","believe","below","beside","besides","between","beyond","biol","both","brief","briefly","but","by","c","ca","came","can","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","could","couldnt","d","date","did","didn't","different","do","does","doesn't","doing","done","don't","down","downwards","due","during","e","each","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","et-al","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","few","ff","fifth","first","five","fix","followed","following","follows","for","former","formerly","forth","found","four","from","further","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","had","happens","hardly","has","hasn't","have","haven't","having","he","hed","hence","her","here","hereafter","hereby","herein","heres","hereupon","hers","herself","hes","hi","hid","him","himself","his","hither","home","how","howbeit","however","hundred","i","id","ie","if","i'll","im","immediate","immediately","importance","important","in","inc","indeed","index","information","instead","into","invention","inward","is","isn't","it","itd","it'll","its","itself","i've","j","just","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","m","made","mainly","make","makes","many","may","maybe","me","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","more","moreover","most","mostly","mr","mrs","much","mug","must","my","myself","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","no","nobody","non","none","nonetheless","noone","nor","normally","nos","not","noted","nothing","now","nowhere","o","obtain","obtained","obviously","of","off","often","oh","ok","okay","old","omitted","on","once","one","ones","only","onto","or","ord","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","owing","own","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","re","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","s","said","same","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","she","shed","she'll","shes","should","shouldn't","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","so","some","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","such","sufficiently","suggest","sup","sure"]

# only select the text out of each tweet object
tweet_texts = sqlContext.sql("SELECT text FROM tweets")

# takes a unicode string (a tweet) and removes stopwords, then returns a string without them
def removeStopWords(tweet, stopword_list):
    tweet_list = tweet.split()
    return " ".join([x for x in tweet_list if x.lower() not in stopword_list])

# tweets without stopwords
tw_wosws = tweet_texts.map(lambda tw: removeStopWords(tw[0], stopwords))

# make key-value tuples from a text, with values always being 1
# this makes the tweet be ready for combineByKey
def wordsFromText(tweet):
    word_list = tweet.split()
    words = []
    for i in word_list:
        words.append( (i.lower(), 1) )
    return words

# word count for each of the words in the corpus
word_count = tw_wosws.flatMap(lambda tweet: wordsFromText(tweet)).reduceByKey(lambda a, b: a+b).collect()

# create a list out of the rare words in the list
rare_words = [x[0] for x in word_count if x[1] <= 1]

# THIS COULD BE DEPRECATED AND USE THE gensim.dictionary.filter_extremes function instead!
def removeRareWords(tweet, rare_words_list):
    tweet_list = tweet.split()
    return " ".join([x for x in tweet_list if x.lower() not in rare_words_list])


def genCommonWordsList(tweets, amount):
    common_words_list = []
    tw_sorted = sorted(tweets, key=lambda x: x[1])
    cursor = 0
    while cursor <= amount:
        common_words_list.append(tw_sorted[-cursor][0])
        cursor = cursor + 1
    return common_words_list

common_words = genCommonWordsList(word_count, 50)

# THIS COULD BE DEPRECATED AND USE THE gensim.dictionary.filter_extremes function instead!
def removeCommonWords(tweet, common_words_list):
    tweet_list = tweet.split()
    return " ".join([x for x in tweet_list if x.lower() not in common_words_list])

# now we have the tw_woswarw (tweets without stopwords and rare words) list of tweets as lists of strings ready to be processed by gensim
# also, remove special characters


def normalizeAndSplit(tweet):
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
            ascii_tweet = tweet.decode().rstrip('\n')
        # drop special characters
        nospec_tweet = re.sub('[^a-zA-Z\d\s:/]+', '', ascii_tweet)
        # lowercase and split
        norm_tweet = nospec_tweet.lower()
        # if nothing remains, return empty array
        if len(norm_tweet) > 0:
            return norm_tweet.split()
        else:
            return []


tw_woswarw = tw_wosws.map(lambda tweet: removeRareWords(tweet, rare_words)).map(lambda tweet: removeCommonWords(tweet, common_words)).map(lambda tweet: normalizeAndSplit(tweet) )

texts = tw_woswarw.collect()

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

#dictionary.save('/tmp/tweets.dict')
#corpora.MmCorpus.serialize('/tmp/tweets.mm', corpus)

#dictionary = corpora.Dictionary.load('/tmp/tweets.dict')
#corpus = corpora.MmCorpus('/tmp/tweets.mm')

def outputTopicDistributions(num_topics):
    # extract LDA topics
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, update_every=0, passes=20)
    # Now that the topic model is built (variable named lda),
    # next we want to iterate over each tweet (after normalizing the tweet)
    # and see in which topic that specific tweet goes to, and then to produce tuples of tweet ID and topic ID
    def toVector(tweet_text):
        return dictionary.doc2bow(normalizeAndSplit(tweet_text))
    def printProbVect(vec):
         return lda[toVector(vec)]
    tweet_texts = sqlContext.sql("SELECT id_str as id, text FROM tweets")
    distros = tweet_texts.map(lambda tw: (tw.id, printProbVect(tw.text), tw.text ) )
    # (u'7 Simple Website Changes To Boost Your..', [(9, 0.11529808102540036), (74, 0.1205543075072763), (81, 0.65636983368954605)]
    def formatDistroCSV(tweet_distro):
        output_list = []
        # append the tweet ID first in double quotes to emphasize it's a string.
        output_list.append("\"id_"+str( tweet_distro[0])+"\"")
        # next, generate the tweet long-form, untouched text, but first transform it into ASCII if it's unicode
        # but append only last for prettier formatting
        ascii_tweet = ''
        if type(tweet_distro[2]) is bytes:
            print("This is bytes: ", tweet, " and we need to change it.")
            raise
        else:
            # yes
            #ascii_tweet = unicodedata.normalize('NFKD', tweet).encode('ascii','ignore')
            if type(tweet_distro[2]) is str:
                ascii_tweet = tweet_distro[2]
            else:
                ascii_tweet = tweet_distro[2].decode()
        ascii_tweet = ascii_tweet.rstrip('\n')
        #ascii_tweet = unicodedata.normalize('NFKD', tweet_distro[2]).encode('ascii','ignore').rstrip('\n')
        # in order to not have to do this in each iteration
        # of the populating of the non-sparse list, let's do it now
        sparse_keys_list = []
        for i in tweet_distro[1]:
            sparse_keys_list.append(i[0])
        # iterate over the sparse distro list and create non-sparse list
        for i in range(num_topics):
            # each iteration represents one topic group in the topic model with 100 topics by default
            # remember python zero-index vs. 1-index topic ID's
            # i represents the topic number
            if i+1 in sparse_keys_list:
                # j represents the different probability distribution list items which are tuples ([int]index, [float]probability)
                for j in tweet_distro[1]:
                    if i+1 == j[0]:
                        output_list.append(j[1])
            else:
                output_list.append(0)
        # append the tweet last
        # -----> CANNOT GET THE LINE BREAKS AWAY EVEN WITH THIS :(((
        output_list.append( re.sub('[^a-zA-Z]+', '-', ascii_tweet) )
        #finally, return the whole shebang
        return ', '.join(str(v) for v in output_list)
    # and we're done!
    distro_rows = distros.map(lambda t: formatDistroCSV(t))
    distro_rows.saveAsTextFile('/Users/timo/Code/with_'+str(num_topics))


distro_nums = [5,10,25,50,100]
for i in distro_nums:
    outputTopicDistributions(i)
