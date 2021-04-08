from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
from numpy import array
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
import json
import os

# process texts
def process_review(sentence):
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.lower()
    return sentence
data = []
# load data from json data file
with open("Digital_Music_5.json", "r") as read_file:
    for line in read_file:
        data.append(json.loads(line))
reviews = []
rating = []
# store review and rating from data
for line in data:
    reviews.append(process_review(line['reviewText']))
    rating.append(line['overall'])


# load glove dictionary file and store them into dictionary
dic = dict()
glove_file = open("glove.6B.100d.txt","r", encoding ="utf8")
for vector in glove_file:
    data = vector.split()
    word = data[0]
    word_vector = np.array(data[1:], dtype ="float32")
    dic[word] = word_vector

review_matrix = []
# modifying data so that there will be 5460 positve and 5460 negative
new_review = []
new_rating = []
i,j = 0,0
max = 0
for x in range(len(reviews)):
    words = process_review(reviews[x]).split(" ")
    if rating[x] == 5 and i < 5460 and len(words) <= 400:
        new_review.append(reviews[x])
        new_rating.append(rating[x])
        i += 1
        if max < len(words):
            max = len(words)
    elif j < 5460 and len(words) <= 400 and (rating[x] == 1 or rating[x] == 2):
        new_review.append(reviews[x])
        new_rating.append(rating[x])
        j += 1
        if max < len(words):
            max = len(words)
world_net = reviews
reviews = new_review
rating = new_rating
# set up the matrix for CNN model
matrix = []
for x in reviews:
    words =  process_review(x).split(" ")
    i = 0
    matrix_word = np.zeros((400,100))
    for word in words:
        if word in dic.keys():
            matrix_word[i] = np.array(dic[word])
        i += 1
    matrix.append(matrix_word)
matrix = np.array(matrix)

# set labels
rating = np.array(list(map(lambda x: 1 if x == 5 else 0, rating)))
labels = []
i,j = 0,0
for x in range(len(rating)):
    label = np.zeros(2)
    if rating[x] == 1:
        label[1] = 1
        i += 1
    else:
        label[0] = 1
        j += 1
    labels.append(label)

# to see the size of the dataset
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)
print(X.toarray().shape)
# testing purpose
tokenizer = Tokenizer()
tokenizer.fit_on_texts(reviews)
reviews = tokenizer.texts_to_sequences(reviews)


def get_data():
    return matrix,np.array(labels)


#vectorizer = CountVectorizer()
#X = vectorizer.fit_transform(reviews)
