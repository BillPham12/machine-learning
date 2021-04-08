import numpy as np
import re
from numpy import array
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
import json
from matplotlib import pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer

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
dic = dict()
rating_dic = dict()
for line in data:
    words = process_review(line['reviewText']).split(" ")
    if len(words) not in dic.keys():
        dic[len(words)] = 1
    else:
        dic[len(words)] += 1
    reviews.append(process_review(line['reviewText']))

    if line['overall'] not in rating_dic.keys():
        rating_dic[line['overall']] = 1
    else:
        rating_dic[line['overall']] += 1
    rating.append(line['overall'])
# 5801 negative and 35580 positive
print("The summary of rating from the dataset",rating_dic)
# storing dictionary
dictionary = []
for element in dic:
    dictionary.append((element,dic[element]))
dictionary.sort(key = lambda x: x[1],reverse =True)
reviews_length = []
frequency = []
for x in dictionary:
    reviews_length.append(x[0])
    frequency.append(x[1])

print("There are", len(reviews_length), "different in reviews' size")
# generate graph lengths of reviews
generate_frequency = False
if generate_frequency:
    plt.bar(reviews_length,frequency,color = "green",label = "word frequency")
    plt.legend()
    plt.title("REVIEWS'LENGTH FREQUENCY")
    plt.xlabel("The reviews' length", fontsize=10)
    plt.ylabel('The number of reviews', fontsize=10)
    plt.show()


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
reviews = new_review
world_net = reviews
rating = new_rating
print("The total number of reviews after modifying",len(reviews))
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)
print("The total number of different words from dataset's dictionary",X.toarray().shape[1])
# using token to modify text into array index

# finding adjectives from tokenizer based on world_net dictionary.
find_sentiment = False
if find_sentiment:
    t = Tokenizer()
    t.fit_on_texts(world_net)
    dic_tokenizer = t.word_counts
    sentiment = dict()
    for word in dic_tokenizer:
        open = os.popen("C:\\Users\\billpham\\Desktop\\WordNet\\bin\\wn "+word+" -n# -searchtype -over").read()
        if "adj" in open:
            sentiment[word] = 1
    new_data = open("new_data.txt","r")
    for x in dic:
        new_data.write(str(x))
    exit()
# create the adjective data file
generate_data_file = False
if generate_data_file:
    new_data = open("new_data.txt","r").readlines()
    for x in new_data:
        new_words = x.split(": 1, '")
        new_words[0] = 'hard'
        new_words[len(new_words)-1] = 'untraveled'
        words_data = open("adjective_words.txt","w+")
        for x in range(1,len(new_words)-1):
            new_words[x] = new_words[x][:-1]
            words_data.write(str(new_words[x]+"\n"))


# generate adjective vector
new_data = open("adjective_words.txt","r")
adj_words = []
for x in new_data.readlines():
    adj_words.append(str(x))


#fitting adjective words to tokenizer
t =  Tokenizer()
t.fit_on_texts(adj_words)
# vector size of 9587
world_net = t.texts_to_sequences(world_net)
special_data  = []

for x in range(len(world_net)):
    vector = np.zeros(9587,dtype = "float32")
    for index in world_net[x]:
        vector[index-1] += 1
    special_data.append(vector)

# print out lengths of the list
print(len(rating))
print(len(new_review))

# full data: 107680 words
print("Done modify data")
# size of input vector: 37950 words to num_words
def get_data(num_words):
    reviews = new_review
    tokenizer = Tokenizer(num_words = num_words)
    tokenizer.fit_on_texts(reviews)
    reviews = tokenizer.texts_to_sequences(reviews)
    size = 5460*2
    data = []
    labels = []
    f, j = 0, 0
    for x in range(size):
        vector = np.zeros(num_words, dtype = "float32")
        label = np.zeros(2,dtype = "float32")
        for y in reviews[x]:
            vector[y] += 1
        if rating[x]  == 5:
            label[1] = 1
            data.append(vector)
            labels.append(label)
            f+=1
        else:
            label[0] = 1
            data.append(vector)
            labels.append(label)
            j+=1

    return np.array(data),np.array(labels)

# size of input vector: 37950 words to 9537 adjective words
def get_special_data():
    return np.array(special_data)
