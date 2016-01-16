hdp = twpr.doHDP(corpus, dictionary)

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
