# lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, update_every=0, passes=20)
# num_topics = 25

lda.show_topics(num_topics=num_topics)
