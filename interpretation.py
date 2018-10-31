# -*- coding: utf-8 -*-

"""
	Created by Shaheen Syed

	For reference articles see:
	Syed, S., Borit, M., & Spruit, M. (2018). Narrow lenses for capturing the complexity of fisheries: A topic analysis of fisheries science from 1990 to 2016. Fish and Fisheries, 19(4), 643–661. http://doi.org/10.1111/faf.12280
	Syed, S., & Spruit, M. (2017). Full-Text or Abstract? Examining Topic Coherence Scores Using Latent Dirichlet Allocation. In 2017 IEEE International Conference on Data Science and Advanced Analytics (DSAA) (pp. 165–174). Tokyo, Japan: IEEE. http://doi.org/10.1109/DSAA.2017.61
	Syed, S., & Spruit, M. (2018a). Exploring Symmetrical and Asymmetrical Dirichlet Priors for Latent Dirichlet Allocation. International Journal of Semantic Computing, 12(3), 399–423. http://doi.org/10.1142/S1793351X18400184
	Syed, S., & Spruit, M. (2018b). Selecting Priors for Latent Dirichlet Allocation. In 2018 IEEE 12th International Conference on Semantic Computing (ICSC) (pp. 194–202). Laguna Hills, CA, USA: IEEE. http://doi.org/10.1109/ICSC.2018.00035
	Syed, S., & Weber, C. T. (2018). Using Machine Learning to Uncover Latent Research Topics in Fishery Models. Reviews in Fisheries Science & Aquaculture, 26(3), 319–336. http://doi.org/10.1080/23308249.2017.1416331

"""


"""
	IMPORT MODULES
"""
import logging, sys, re, matplotlib
from operator import itemgetter
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd
import numpy as np


"""
	IMPORT CLASSES
"""
from database import MongoDatabase
from helper_functions import *



class Interpretation():

	def __init__(self):

		logging.info('Initialized {}'.format(self.__class__.__name__))

		# instantiate database
		self.db = MongoDatabase()

		# define plot folder
		self.output_folder = 'output/'
		create_dir(self.output_folder)

		# set utf8 encoding
		reload(sys)
		sys.setdefaultencoding('utf8')

	
	def infer_document_topic_distribution(self, K = 10, dir_prior = 'auto', random_state = 42, num_pass = 15, iteration = 200, top_n_words = 10, 
										models_folder = 'models', file_folder = 'files'):


		"""
			Infer the document topic distribition per publication. The LDA model shows us the word probabilies per topic, but we also want to know what
			topics we find within each document. Here we infer such document-topic distribution and save it to the databse so we can use it later
			to plot some interesting views of the corpus

			Values for K, dir_prior, random_state, num_pass and iteratrion will become visible when plotting the coherence score. Use the model that 
			achieved the highest coherence score.

			models_folder = where the LDA models were stored
			file_folder = where the corpus and dictionary are stored
		"""

		logging.info('Start {}'.format(sys._getframe().f_code.co_name))

		# read dictionary and corpus
		dictionary, corpus = get_dic_corpus(file_folder)

		# load LDA model according to parameters
		model = load_LDA_model('{}/{}/{}/{}/{}/{}/lda.model'.format(models_folder, K, dir_prior, random_state, num_pass, iteration))
		
		# load docs
		D = self.db.read_collection(collection = 'publications_raw')

		# loop through all the documents to infer document-topics distribition
		for i, d in enumerate(D):

			# check if tokens are present; in case some documents couldn't properly be tokenized during pre-processing phase
			if d.get('tokens') is not None:

				# print to console
				print_doc_verbose(i, D.count(), d['journal'], d['year'], d['title'])

				# create bag of words from tokens
				bow = model.id2word.doc2bow(d['tokens'])

				# infer document-topic distribution
				topics = model.get_document_topics(bow, per_word_topics = False)

				# convert to dictionary: here we convert the topic number to string because mongodb will complain otherwise
				# you will get a message that documents can only have string keys
				dic_topics = {}
				for t in topics:
					dic_topics[str(t[0])] = float(t[1])
					
				# create a new document to add to the database, this time in a different collection
				insert_doc = {'journal': d['journal'], 'year' : d['year'], 'title' : d['title'], 'topics' : dic_topics}
				
				# save insert_doc to database within publications collection
				self.db.insert_one_to_collection('publications', insert_doc)


	def get_document_title_per_topic(self):


		"""
			Get document title per topic
			Here we obtain the publication title of the most dominant topic within that publication
			Most dominant topic is the topic proportion that is the largest
			So if document has topic A = 10%, B = 30%, and C = 60%, then C is the dominant topic
			We can use the titles for the dominant topics to get insights into the label of that topic 
		"""

		logging.info('Start {}'.format(sys._getframe().f_code.co_name))

		# load docs
		D = self.db.read_collection(collection = 'publications')

		# empty list where we can append publication titles to
		titles = []

		# loop trough all the docs
		for i, d in enumerate(D):
			
			# print to console
			print_doc_verbose(i, D.count(), d['journal'], d['year'], d['title'])
			
			# get the dominant topic
			dominant_topic = max(d['topics'].iteritems(), key = itemgetter(1))
			# get the topic ID and percentage
			dominant_topic_id, dominant_topic_percentage = dominant_topic[0], dominant_topic[1]

			# append to list
			titles.append([d['year'], d['title'], d['journal'], dominant_topic_id, dominant_topic_percentage])
		
		# save to CSV
		saveCSV(titles, 'titles-to-topics', folder = self.output_folder)


	def plot_topics_over_time(self, plot_save_name = 'topics-over-time.pdf'):

		"""
			Plot cumulative topic distribution over time
		"""

		logging.info('Start {}'.format(sys._getframe().f_code.co_name))

		# load docs
		D = self.db.read_collection(collection = 'publications')

		# create dictionary where we can obtain the topic distribution per year
		year_to_topics = get_year_to_topics(D)

		# calculate the cumulative topic distribution: basically the average distribution per year
		year_to_cum_topics = get_year_to_cum_topics(year_to_topics)

		# convert dictionary to pandas dataframe
		df = pd.DataFrame.from_dict(year_to_cum_topics)
		
		# create the plot
		fig, axs = plt.subplots(2,5, figsize=(15, 10))
		axs = axs.ravel()

		# loop over each row of the dataframe
		for index, row in df.iterrows():

			# get year values
			x = df.columns.values.tolist()
			# get topic proportions
			y = row.tolist()

			# add to plot
			axs[index].plot(x, y, 'o--', color='black', linewidth=1, label="Topic prevalence")
			axs[index].set_title(get_topic_label(index), fontsize=14)
			axs[index].set_ylim([0,0.4])

		# save plot
		plt.tight_layout()
		plt.savefig(self.output_folder + plot_save_name)
		plt.close()

	def plot_topics_over_time_stacked(self, plot_save_name = 'topics-over-time-stacked.pdf'):

		"""
			Plot topics over time stacked
		"""

		logging.info('Start {}'.format(sys._getframe().f_code.co_name))

		# load docs
		D = self.db.read_collection(collection = 'publications')

		# create dictionary where we can obtain the topic distribution per year
		year_to_topics = get_year_to_topics(D)

		# calculate the cumulative topic distribution: basically the average distribution per year
		year_to_cum_topics = get_year_to_cum_topics(year_to_topics)

		# convert dictionary to pandas dataframe
		df = pd.DataFrame.from_dict(year_to_cum_topics)

		# transpose dataframe
		df = df.transpose()

		# change column headers into topic labels
		df.columns = [get_topic_label(x) for x in df.columns.values]
		
		# plot the dataframe
		ax = df.plot(figsize = (15, 8), kind = 'area', colormap='Spectral_r', rot = 45, grid = False)
		# set values for x-axis
		plt.xticks(df.index)
		# limit the x-axis
		plt.xlim(min(df.index), max(df.index))
		# limit the y-axis
		plt.ylim(0,1)
		# get the legend
		handles, labels = ax.get_legend_handles_labels()
		# position it right of the figure
		plt.legend(reversed(handles), reversed(labels), loc = 'right', bbox_to_anchor=(1.35, 0.50), ncol=1, fancybox=False, shadow=False, fontsize=16)
		
		# save plot
		plt.tight_layout()
		plt.savefig(self.output_folder + plot_save_name)
		plt.close()


	def plot_topic_co_occurrence(self, plot_save_name = 'topic-co-occurrence.pdf'):

		"""
			Plot topic co-occurrence
		"""

		logging.info('Start {}'.format(sys._getframe().f_code.co_name))

		# load docs
		D = self.db.read_collection(collection = 'publications')

		# create empty dictionary where we can store the dominant topic id and remaining other proportions
		dominant_id_to_topics = {}

		for d in D:

			# sort topics and create list
			topics = [value for key, value in sorted(d['topics'].iteritems(), key=lambda x: int(x[0]))]

			# get max topix id
			max_topic_id = topics.index(max(topics))

			# check if topic ID key already created
			if max_topic_id not in dominant_id_to_topics:
				dominant_id_to_topics[max_topic_id] = []

			dominant_id_to_topics[max_topic_id].append(topics)

		# create empty dictionary where we can have the cumulative topic distribution per dominant topic ID
		dominant_id_to_cum_topics = {}
		for k, v in dominant_id_to_topics.iteritems():

			# calculate mean and add to dictionary
			dominant_id_to_cum_topics[k] = np.mean(np.array(v), axis=0) * 100. 


		# convert dictionary to pandas dataframe
		df = pd.DataFrame.from_dict(dominant_id_to_cum_topics)

		# change column headers into topic labels
		df.columns = [get_topic_label(x) for x in df.columns.values]
		df.index = [get_topic_label(x) for x in df.index.values]

		# create max column
		df['max'] = 0.

		# keep track of new index
		new_index = []

		# add max column so we can sort on it later
		for index, row in df.iterrows():

			# add value to max column
			df['max'][index] = max(row)
			# make self co-occurrence zero
			df[index][index] = 0.0

			# add new index names to tracker so we can rename it later
			new_index.append('{} ({}%)'.format(index, round(max(row), 2)))
			
		# update index name
		df.index = new_index

		# sort by max column
		df = df.sort_values(by=['max'], ascending=False)

		# delete max column
		df = df.drop(['max'], axis=1)

		# sort based on column totals
		df = df.reindex(sorted(df.columns), axis=1)

		# plot the heatmap
		ax = sns.heatmap(df, cmap = "Blues", annot = True, vmin = 0., vmax = 10., square = True, annot_kws = {"size": 11},
							fmt = '.1f', mask= df <= 0.0, linewidths = .5, cbar = False)

		# adjust the figure somewhat
		ax.xaxis.tick_top()
		plt.yticks(rotation=0)
		plt.xticks(rotation=90, ha = 'left') 
		fig = ax.get_figure()
		fig.set_size_inches(19, 6)

		# save figure
		fig.savefig(self.output_folder + plot_save_name , bbox_inches='tight')


""" 

internal helper function

"""

def print_doc_verbose(i, total, journal, year, title):

	# console output
	logging.debug('processing file: {}/{}'.format(i+1,total))
	logging.debug('journal : {}'.format(journal))
	logging.debug('year : {}'.format(year))
	logging.debug('title : {}'.format(title))


def get_year_to_topics(D):


	"""
		Create dictionary where we group all the document-topic distributions per year 
	"""

	# create the empty dictionary
	year_to_topics = {}

	# loop trough documents and create the dictionary
	for d in D:

		# check if year key already created
		if int(d['year']) not in year_to_topics:
			year_to_topics[int(d['year'])] = []

		# sort topics and create list
		topics = [value for key, value in sorted(d['topics'].iteritems(), key=lambda x: int(x[0]))]
		
		# add topic distribution to year key
		year_to_topics[int(d['year'])].append(topics)

	return year_to_topics


def get_year_to_cum_topics(year_to_topics):

	"""
		Create dictionary where we obtain the cunulative document-topic distributions per year 
		cumulative document-topic distributions are mean values for each topic
	"""

	# create empty dictionay
	year_to_cum_topics = {}

	# loop over year_to_topics dictionary are calculate mean of topics
	for k, v in year_to_topics.iteritems():

		# calculate the column mean
		mean_topics = np.mean(np.array(v), axis=0)
		
		# add to dictionary so we can obtain it later on
		year_to_cum_topics[k] = mean_topics
	

	return year_to_cum_topics
