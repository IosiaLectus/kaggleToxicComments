#!/usr/bin/python

# Use the sklearn and nltk to classify toxic comments.

# Import useful packages
import pandas, nltk, re

import numpy as np
import csv as csv

from matplotlib import pyplot
from nltk import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy

# Define a lexical diveristy function
def lexical_diversity(text):
	if len(text) == 0:
		return 0
	return len(set(text))/len(text)


def contains_one(s,lst):
	for w in lst:
		if re.search(w,s,re.IGNORECASE):
			return 1
	return 0

# Lists of potentially relevant words
vulgarities = ['cock','dick','cunt','ass','shit','fuck','bitch']
slurs = [' fag',' nigg','slut']
insults = ['stupid','moron']
identities = ['jew','mexican','gay','homosexual']
violence = ['murder','stab','shoot','kill','die','rape']
frequent_long_words = ['nigger', 'faggot', 'stupid', 'yourself','fuck','going','fat','jew','suck','die','ass','kill','shit','consensus','adding','check','cock','spanish','wales','murder']

# Create a set of possibly relevant features
def decorate(trn):

	# Tokenize comments
	trn['comment_text_tokenized'] = trn.apply(lambda x: word_tokenize(x['comment_text'].decode('utf-8')), axis=1)

	# Create comment length column
	trn['length'] = trn.apply(lambda x: len(x['comment_text_tokenized']), axis=1)

	# Create lexical diversity column
	trn['lexical_diversity'] = trn.apply(lambda x: lexical_diversity(x['comment_text_tokenized']), axis=1)

	# Check for words in 'vulgarities' list
	trn['contains_vulgarity'] = trn.apply(lambda x: contains_one(x['comment_text'], vulgarities), axis=1)
	
	# Check for words in 'slurs' list
	trn['contains_slur'] = trn.apply(lambda x: contains_one(x['comment_text'], slurs), axis=1)

	# Check for words in 'insluts' list
	trn['contains_insult'] = trn.apply(lambda x: contains_one(x['comment_text'], insults), axis=1)

	# Check for words in 'identities' list
	trn['contains_identity'] = trn.apply(lambda x: contains_one(x['comment_text'], identities), axis=1)

	# Check for words in 'violence' list
	trn['contains_violence'] = trn.apply(lambda x: contains_one(x['comment_text'], violence), axis=1)

	# Check for each of the frequent words
	for w in frequent_long_words:
		trn['fw_'+w] = trn.apply(lambda x: contains_one(x['comment_text'], [w]), axis=1)


# Now look at which features differentiate between categories
def check_features(trn):
	# Create texts representing all toxic comments etc. as well as all comments that are 'none of the above'. Also count the number of each type

	labels = ['total','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'vanilla']
	df_dict = {};

	df_dict['total'] = trn
	df_dict['vanilla'] = trn.query('toxic==0 and severe_toxic==0 and obscene==0 and threat==0 and insult==0 and identity_hate==0')

	for label in labels[1:-1]:
		df_dict[label] = trn.query(label + '==1')

	# Store averages of features for each category
	feature_columns = trn.columns[9:]
	summary_df = pandas.DataFrame(columns = feature_columns, index = labels)

	# Concat the individual comments in each category to a large text whose statistics can be analyzed
	subtexts = {}
	# lower case version of the above
	subtexts_lower = {}
	# County the number of comments in each category
	counts = {}
	# For each category, get the frequency distribution
	frequency_distributions = {}
	# Convert frequencies from absolute counts to fractions of occurences. Not yet implemented
	frac_freq = {}
	# Frequency distribution over words over some minimum length (currently 4)
	big_word_freq = {}
	# Frequency distribution over words not common in the total set
	freq2 = {}
	# Frequency distribution over words not common in the 'toxic' set
	freq3 = {}

	# Fill out the above dictionaries
	for label in labels:

		df = df_dict[label]

		counts[label] = 0
		subtexts[label] = []
		subtexts_lower[label] = []
		for index, row in df.iterrows():
			subtexts[label] += row['comment_text_tokenized']
			subtexts_lower[label] += [w.lower() for w in row['comment_text_tokenized']]
			counts[label] += 1
		frequency_distributions[label] = nltk.FreqDist(subtexts_lower[label])
		big_word_text = [w for w in subtexts_lower[label] if len(w)>4]
		big_word_freq[label] = nltk.FreqDist(big_word_text)

		summary_df.loc[label,'counts'] = counts[label]
		for c in feature_columns:
			summary_df.loc[label,c] = df[c].mean()

	purge_list = [tp[0] for tp in frequency_distributions['total'].most_common(300)]
	purge_list2 = [tp[0] for tp in frequency_distributions['toxic'].most_common(300)]
	for label in labels:
		text2 = [w for w in subtexts_lower[label] if not w in purge_list]
		freq2[label] = nltk.FreqDist(text2)
		text3 = [w for w in subtexts_lower[label] if not w in purge_list2]
		freq3[label] = nltk.FreqDist(text3)

	print

	for label in labels:
		df = df_dict[label]
		print label
		print df['toxic'].mean()
		print freq3[label].most_common(10)
		print

	# Save summary_df

	summary_df.to_csv("summary.csv")

def main():
	# Read in the test and train files

	#train = pandas.read_csv("small_train.csv")
	train = pandas.read_csv("train.csv")
	test = pandas.read_csv("test.csv")

	# Use a smaller dataframe for testing code
	#small_train = train.head(250)

	# Save modified csv
	#small_train.to_csv("small_train.csv")

	# Check the efficacy of the features
	#check_features(train)

	# labels we are classifying on
	labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

	# Create prediction df
	predict = pandas.DataFrame(columns = labels, index = test['id'])
	print
	print 'predict created'

	# Traning targets
	target_lst = [train[label] for label in labels]

	# Define features
	decorate(train)
	print
	print 'train decorated'
	decorate(test)
	print
	print 'test decorated'

	## Save modified csv
	train.to_csv("train_dec.csv")

	# Drop columns not used in training
	train_features = train.drop(['comment_text','comment_text_tokenized','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','id'],axis=1)
	test = test.drop(['comment_text','comment_text_tokenized','id'],axis=1)

	# Create random forest
	my_forest_lst = [RandomForestClassifier(n_estimators=100) for label in labels]

	labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
	
	# train the classifiers
	for i in range(0,len(labels)):
		my_forest_lst[i].fit(train_features,target_lst[i])
		print
		print ("forest for " + labels[i] + " trained")
		predict[labels[i]] = my_forest_lst[i].predict_proba(test)[:,1]
	
	print
	predict.to_csv("prediction.csv")
	print "prediction saved"
	print


if __name__ == '__main__':
	main()
