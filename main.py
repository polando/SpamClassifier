import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            lines = []

            f = io.open(path, 'r', encoding='latin-1')
            for line in f:
                lines.append(line)
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)


data = DataFrame()

data = data.append(dataFrameFromDirectory('.../enron1/spam', 'spam'))
data = data.append(dataFrameFromDirectory('.../enron1/ham', 'ham'))

classifier = MultinomialNB()
vectorizer = CountVectorizer()

counts = vectorizer.fit_transform(data['message'].values)
tragets = data['class'].values

classifier.fit(counts,tragets)

example = ['free drugs now','hello I will come tomorrow', 'talk more using this secret code']

ex_counts= vectorizer.transform(example)

pred = classifier.predict(ex_counts)
print(pred)
