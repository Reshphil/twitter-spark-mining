path = "/Users/timo/Code/with_25" # TSV file format
#path = "/Users/timo/Ruby/GetTweets/stored_tweets/*"
raw = sc.textFile(path)
data = raw.map(lambda x: x.split("\t"))

headers = data.map(lambda x: x[0]).collect()
values = data.map(lambda x: x[1:-1]).collect()
rows = data.map(lambda x: x).collect()

import numpy as np
# import re

# rows = []
# for i in range(len(headers)):
#     arr = np.array(values[i])
#     np.array(
#         arr,
#         dtype='float'
#     )

# colnames = [('name', 'object')]
# cols = [('topic'+str(i+1), 'float') for i in range(len(data.take(1)[0])-1)]
# colnames.extend(cols)

val_mat = np.matrix( values, dtype='float' )

mat = np.matrix(rows)
mat[np.logical_or.reduce([mat[:,2] >= [0.75]])]

# daily_prices = np.array(
#     [
#         (4,3,3,1),
#         (5,4,3,6),
#         (6,3,2,7),
#         (3,9,7,4),
#         (8,4,6,3),
#         (8,3,3,9)],
#     dtype=[('MSFT','float'),('CSCO','float'),('GOOG','float'),('F','float') ]
#     )
