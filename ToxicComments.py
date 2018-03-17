#!/usr/bin/python3

# Use the sklearn and nltk to classify toxic comments.

# Import useful packages
import pandas, nltk, re, math

import numpy as np
import csv as csv

from nltk import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from copy import deepcopy

def mydiv(x,y):
	if y==0:
		return 0
	return x/y

# Define a lexical diveristy function
def lexical_diversity(text):
	return mydiv(len(set(text)),len(text))

# Get the Signal over background, but penalizing statistical error
def SOB(p,n):
	if n==0:
		return 0
	er = math.sqrt(p*(1-p))/n
	p = p - er
	return p/(1-p + 0.0000001)


def contains_one(s,lst):
	if not isinstance(s,str):
		s = s.decode('utf-8')
	for w in lst:
		if re.search(re.escape(w),s,re.IGNORECASE):
			return 1
	return 0

def list_count(item,lst):
	tally = [1 for x in lst if x==item]
	return len(tally)

# Lists of potentially relevant words
vulgarities = ['cock','dick','cunt','ass','shit','fuck','bitch','piss']
slurs = [' fag',' nigg','slut']
insults = ['stupid','moron']
identities = ['jew','mexican','gay','homosexual','vietnam','chinese']
violence = ['murder','stab','shoot','kill','die','rape']
frequent_long_words = ['nigger', 'faggot', 'stupid', 'yourself','fuck','going','fat','jew','suck','die','ass','kill','shit','consensus','adding','check','cock','spanish','wales','murder','mother','sex', 'I will','!', 'is a','!!','fat jew','huge faggot', 'nigger nigger', 'fuck you', 'die die']

# Create a set of possibly relevant features
def decorate(trn):

	# Tokenize comments
	trn['comment_text_tokenized'] = trn.apply(lambda x: word_tokenize(x['comment_text']), axis=1)

	# Tag words
	trn['comment_text_tags'] = trn.apply(lambda x: [tpl[1] for tpl in nltk.pos_tag(x['comment_text_tokenized'])], axis=1)

	# Noun frequency
	trn['noun_freq'] = trn.apply(lambda x: mydiv((list_count('NN',x['comment_text_tags']) + list_count('NNP',x['comment_text_tags'])),len(x['comment_text_tokenized'])), axis=1)

	# Adjective frequency
	trn['adjective_freq'] = trn.apply(lambda x: mydiv(list_count('JJ',x['comment_text_tags']),len(x['comment_text_tokenized'])), axis=1)

	# Verb frequency
	trn['verb_freq'] = trn.apply(lambda x: mydiv((list_count('VB',x['comment_text_tags']) + list_count('VBP',x['comment_text_tags'])),len(x['comment_text_tokenized'])), axis=1)

	# Adjective frequency
	trn['punct_freq'] = trn.apply(lambda x: mydiv(list_count('.',x['comment_text_tags']),len(x['comment_text_tokenized'])), axis=1)

	# Adjective frequency
	trn['personal_pronoun_freq'] = trn.apply(lambda x: mydiv(list_count('PRP$',x['comment_text_tags']),len(x['comment_text_tokenized'])), axis=1)

	# Create comment length column
	trn['length'] = trn.apply(lambda x: len(x['comment_text_tokenized']), axis=1)

	# Create comment length column
	trn['avg_word_length'] = trn.apply(lambda x: mydiv(len(x['comment_text']),len(x['comment_text_tokenized'])), axis=1)

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
	feature_columns = trn.columns[10:]
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
	# Bigram frequencies
	bigrams = {}
	# Bigram frequencies
	bigrams2 = {}

	# Fill out the above dictionaries
	for label in labels:

		df = df_dict[label]

		counts[label] = 0
		subtexts[label] = []
		subtexts_lower[label] = []
		for row in df.index.values:
			comment = df.loc[row,'comment_text_tokenized']
			subtexts[label] += comment
			subtexts_lower[label] += [w.lower() for w in comment]
			counts[label] += 1
		frequency_distributions[label] = nltk.FreqDist(subtexts_lower[label])
		big_word_text = [w for w in subtexts_lower[label] if len(w)>4]
		big_word_freq[label] = nltk.FreqDist(big_word_text).most_common(10)
		bigrams[label] = nltk.FreqDist(nltk.bigrams(subtexts_lower[label])).most_common(100)

		summary_df.loc[label,'counts'] = counts[label]
		for c in feature_columns:
			summary_df.loc[label,c] = df[c].mean()

	purge_list = [tp[0] for tp in frequency_distributions['total'].most_common(300)]
	purge_list2 = [tp[0] for tp in frequency_distributions['toxic'].most_common(150)]
	purge_list3 = [tp[0] for tp in bigrams['total']]
	for label in labels:
		text2 = [w for w in subtexts_lower[label] if not w in purge_list]
		freq2[label] = nltk.FreqDist(text2).most_common(10)
		text3 = [w for w in subtexts_lower[label] if not w in purge_list2]
		freq3[label] = nltk.FreqDist(text3).most_common(10)
		bigram_list = [bg for bg in nltk.bigrams(subtexts[label]) if not bg in bigrams[label]]
		bigrams2[label] = nltk.FreqDist(bigram_list).most_common(10)

	# Save summary_df

	summary_df.to_csv("summary.csv")


	# Save frequent word lists to a file

	f = open("important_words.txt",'w')
	f_str = ''
	for label in labels:
		f_str += label + "\n" + str(big_word_freq[label]) + "\n" + str(freq2[label]) + "\n" + str(freq3[label]) + "\n" + str(bigrams2[label]) + "\n\n"
	f.write(f_str)
	f.close()


# Find most important words
def important_word_finder(trn):
	# Create texts representing all toxic comments etc. as well as all comments that are 'none of the above'. Also count the number of each type
	labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

	# Create the corpus of all non-vanilla comments put together
	df_tox = trn.query("toxic==1 or severe_toxic==1 or obscene==1 or threat==1 or insult==1 or identity_hate==1")
	corpus = []
	for row in df_tox.index.values:
		comment = nltk.word_tokenize(df_tox.loc[row,'comment_text'].replace('-',' ').replace('\'','').replace('.',' '))
		corpus += [w.upper() for w in comment]

	print('\nCreated corpus')
	
	# Dictionary which gives the number of times a word occurs in the corpus
	fd = {tpl[0]:tpl[1] for tpl in nltk.FreqDist(corpus).most_common(1000)}

	# The set of words in the corpus
	word_set = fd.keys()

	# Words which are not too rare
	good_words = [w for w in word_set if len(w)>3]

	print('\nDefined word set')

	# A dataframe for storing signal over background info
	df = pandas.DataFrame(index=good_words, columns=labels)

	# For each word, make a column in trn which tells us if the comment contains this word
	trn_ary = trn.values
	lst = trn.columns.values
	col_index = {lst[i]:i for i in range(len(lst))}
	v_contains_one = np.vectorize(lambda vec, lst: contains_one(vec, lst))
	print("\nnp setup complete")
	for w in df.index.values:
		trn[w] = v_contains_one( trn_ary[:,col_index['comment_text']], [w] )
		#trn[w] = trn.apply(lambda x: contains_one(x['comment_text'], [w]), axis=1)


	print('\nCreated extra columns in trn')
	
	# Store signal over background info in df
	for w in df.index.values:
		df2 = trn.query(w+'==1')
		tot_len = len(df2.index.values)
		for label in df.columns.values:
			signal = mydiv(len(df2.query(label+"==1").index.values),tot_len)
			df.loc[w,label] = SOB(signal,tot_len)

	print('\nFilled df')

	# Compute mean
	df['mean'] = df.mean(axis=1)

	df.to_csv("sob.csv")
	
	print('\nSaved sob.csv')
	
	# Get a dictionary which stores the average (over labels) SOB value
	hs_dict = {}	
	for w in df.index.values:
		hs_dict[w] = df.loc[w,'mean']

	# Get a list of 'significant' words
	ret_list = [(k, hs_dict[k]) for k in hs_dict.keys() if (hs_dict[k] > 2)]

	# Write the list to a file
	f = open("significant_words.txt",'w')
	f.write(str(ret_list))
	f.close()

	df.to_csv("word_importance.csv")
	
def main():
	# Read in the test and train files

	#train = pandas.read_csv("small_train.csv")
	train = pandas.read_csv("train.csv")
	#test = pandas.read_csv("test.csv")

	# Use a smaller dataframe for testing code
	#small_train = train.head(250)

	# Save modified csv
	#small_train.to_csv("small_train.csv")

	# labels we are classifying on
	#labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

	# Create prediction df
	#predict = pandas.DataFrame(columns = labels, index = test['id'])
	#print('\npredict created')

	# Traning targets
	#target_lst = [train[label] for label in labels]

	# Define features
	#decorate(train)
	#print('\ntrain decorated')
	#decorate(test)
	#print('\ntest decorated')

	# Save modified csv
	#train.to_csv("train_dec.csv")

	# Check the efficacy of the features
	#check_features(train)
	important_word_finder(train)

	# Drop columns not used in training
	#train = train.drop(['comment_text','comment_text_tokenized','comment_text_tags','id'],axis=1)
	#test = test.drop(['comment_text','comment_text_tokenized','comment_text_tags','id'],axis=1)
	#train_features = train.drop(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],axis=1)

	# Create correlation matrix
	#train.corr().to_csv("correlations.csv")
	#print("\nSaved correlation table")

	'''
	# Create random forest
	my_forest_lst = [RandomForestClassifier(n_estimators=100) for label in labels]

	labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

	importance_str = ''

	# train the classifiers
	for i in range(0,len(labels)):
		my_forest_lst[i].fit(train_features,target_lst[i])
		print()
		print("forest for " + labels[i] + " trained")
		importance_str += (labels[i] + ": " +str(my_forest_lst[i].feature_importances_) + "\n\n")
		predict[labels[i]] = my_forest_lst[i].predict_proba(test)[:,1]

	predict.to_csv("prediction.csv")
	print("\nprediction saved")

	f = open("important_features.txt",'w')
	f.write(importance_str)
	f.close()
	'''
	print("\nDone\n")

if __name__ == '__main__':
	main()
