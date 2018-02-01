#!/usr/bin/python

# Use the sklearn and nltk to classify toxic comments.

# Import useful packages
import pandas, nltk

import numpy as np
import csv as csv

from matplotlib import pyplot

from nltk import word_tokenize

# Read in the test and train files

train = pandas.read_csv("small_train.csv")
#train = pandas.read_csv("train.csv")
#test = pandas.read_csv("test.csv")

# Use a smaller dataframe for testing code
#small_train = train.head(250)

# Save modified csv
#small_train.to_csv("small_train.csv")


# Create a set of possibly relevant features

# Tokenize comments
train['comment_text_tokenized'] = train.apply(lambda x: word_tokenize(x['comment_text'].decode('utf-8')), axis=1)

# Create comment length column
train['length'] = train.apply(lambda x: len(x['comment_text_tokenized']), axis=1)

# Define a lexical diveristy function
def lexical_diversity(text):
	return len(set(text))/len(text)

# create lexical diversity column
train['lexical_diversity'] = train.apply(lambda x: lexical_diversity(x['comment_text_tokenized']), axis=1)


# Now look at which features differentiate between categories

# Create texts representing all toxic comments etc. as well as all comments that are 'none of the above'. Also count the number of each type

labels = ['total','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'vanilla']
df_dict = {};

df_dict['total'] = train
df_dict['vanilla'] = train.query('toxic==0 and severe_toxic==0 and obscene==0 and threat==0 and insult==0 and identity_hate==0')

for label in labels[1:-1]:
	df_dict[label] = train.query(label)

subtexts = {}
counts = {}
mean_diversity = {}
mean_length = {}

for label in labels:
	df = df_dict[label]
	mean_diversity[label] = df['lexical_diversity'].mean()
	mean_length[label] = df['length'].mean()
	counts[label] = 0
	subtexts[label] = []
	for index, row in df.iterrows():
		subtexts[label] += row['comment_text_tokenized']
		counts[label] += 1

print

print mean_length

print 

print mean_diversity

print

print counts

print

# Save modified csv
train.to_csv("train_dec.csv")
