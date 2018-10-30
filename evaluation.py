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
import logging, sys, re
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd


"""
	IMPORT CLASSES
"""
from database import MongoDatabase
from helper_functions import *



class Evaluation():

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

	def calculate_coherence(self, file_folder, models_folder):

		"""
			Calculate the CV coherence score for each of the created LDA models
		"""

		logging.info('Start {}'.format(sys._getframe().f_code.co_name))

		# read dictionary and corpus
		dictionary, corpus = get_dic_corpus(file_folder)

		# load bag of words features of each document from the database
		texts = [x['tokens'] for x in self.db.read_collection('publications_raw')]

		# get path location for models
		M = [x for x in read_directory(models_folder) if x.endswith('lda.model')]

		# read processed models from database
		processed_models = ['{}-{}-{}-{}-{}'.format(x['k'], x['dir_prior'], x['random_state'], x['num_pass'], x['iteration']) for x in self.db.read_collection('coherence')]
		
		# calculate coherence score for each model
		for i, m in enumerate(M):

			logging.info('Calculating coherence score: {}/{}'.format(i+1, len(M)))

			# number of topics
			k = m.split('/')[1]
			# different dirichlet priors
			dir_prior = m.split('/')[2]
			# random initiatilizations
			random_state = m.split('/')[3]
			# passes over the corpus
			num_pass = m.split('/')[4]
			# max iteration for convergence
			iteration = m.split('/')[5]

			logging.info('k: {}, dir_prior: {}, random_state: {}, num_pass: {}, iteration: {}'.format(k, dir_prior, random_state, num_pass, iteration))

			# check if coherence score already obtained
			if '{}-{}-{}-{}-{}'.format(k, dir_prior, random_state, num_pass, iteration) not in processed_models: 
				
				# load LDA model
				model = models.LdaModel.load(m)

				# get coherence c_v score
				coherence_c_v = CoherenceModel(model = model, texts = texts, dictionary = dictionary, coherence='c_v')
				
				# get coherence score
				score = coherence_c_v.get_coherence()
				
				# logging output
				logging.info('coherence score: {}'.format(score))

				# save score to database
				doc = {	'k' : k, 'dir_prior' : dir_prior, 'random_state' : random_state, 'num_pass' : num_pass, 'iteration' : iteration, 'coherence_score' : score}
				self.db.insert_one_to_collection('coherence', doc)

			else:
				logging.info('coherence score already calculated, skipping ...')
				continue

	def plot_coherence(self, min_k = 2, max_k = 20, plot_save_name = 'coherence_scores_heatmap.pdf'):

		"""
			Read coherence scores from database and create heatmap to plot all of them
			
			min_k = lowest number of topics created when creating LDA models. Here 2
			max_k = highest number of topics creted when creating LDA models. Here 20
			plot_save_name = where we save the plots to
		"""

		logging.info('Start {}'.format(sys._getframe().f_code.co_name))

		# read documents from database that contain coherence scores
		D = list(self.db.read_collection(collection = 'coherence'))

		# convert data from document into a list
		data = [[int(x['k']), x['dir_prior'],x['random_state'], x['num_pass'], x['iteration'], x['coherence_score']] for x in D]

		# create empty dataframe where we can store our scores
		df = pd.DataFrame()

		# loop trough values of k parameter and find relevant scores for each grid search combination
		for k in range(min_k, max_k + 1):

			# create dataframe to temporarily store values
			df_temp = pd.DataFrame(index = [k])

			# loop trough the data to obtain only the scores for a specific k value
			for row in sorted(data):
				if row[0] == k:
					df_temp['{}-{}-{}-{}'.format(row[1],row[2],row[3],row[4])] = pd.Series(row[5], index=[k])
			
			# append temporarary dataframe of only 1 k value to the full dataframe 
			df = df.append(df_temp)
		
		# transpose the dataframe
		df = df.transpose()
		
		# plot the heatmap
		ax = sns.heatmap(df, cmap = "Blues", annot = True, vmin = 0.500, vmax = 0.530, square = True, annot_kws = {"size": 11},
							fmt = '.3f', linewidths = .5, cbar_kws = {'label': 'coherence score'})

		# adjust the figure somewhat
		ax.xaxis.tick_top()
		plt.yticks(rotation=0)
		plt.xticks(rotation=0, ha = 'left') 
		fig = ax.get_figure()
		fig.set_size_inches(19, 6)

		# save figure
		fig.savefig(self.output_folder + plot_save_name , bbox_inches='tight')



	def output_lda_topics(self, K = 10, dir_prior = 'auto', random_state = 42, num_pass = 15, iteration = 200, top_n_words = 10, models_folder = 'models'):

		"""
			Create table with LDA topic words and probabilities
			Creates a table of topic words and probabilties + topics in a list format
			
			Values for K, dir_prior, random_state, num_pass and iteratrion will become visible when plotting the coherence score. Use the model that 
			achieved the highest coherence score

			max_words = how many words we want to print per topic. Standard 10 but you can choose as many as you want 
		"""

		logging.info('Start {}'.format(sys._getframe().f_code.co_name))

		# load LDA model according to parameters
		model = load_LDA_model('{}/{}/{}/{}/{}/{}/lda.model'.format(models_folder, K, dir_prior, random_state, num_pass, iteration))
		
		# define empty lists so we can fill them with words		
		topic_table, topic_list = [], []

		# loop trough all the topics found within K
		for k in range(K):

			# create topic header, e.g. (1) TOPIC X
			topic_table.append(['({}) {}'.format(k+1, get_topic_label(k).upper())])
			# add column for word and probability
			topic_table.append(["word", "prob."])


			list_string = ""
			topic_string = ""
			topic_string_list = []

			# get topic distribution for topic k and return only top-N words 
			scores = model.print_topic(k, top_n_words).split("+")
			
			# loop trough each word and probability
			for score in scores:

				# extract score and trimm spaces
				score = score.strip()

				# split on *
				split_scores = score.split('*')

				# get percentage
				percentage = split_scores[0]
				# get word
				word = split_scores[1].strip('"')

				# add word and percentage to table
				topic_table.append([word.upper(), "" + percentage.replace("0.", ".")])
				# add word to list table
				list_string += word + ", "

			# add empty line for the table
			topic_table.append([""])
			# add topic words to list
			topic_list.append([str(k+1), list_string.rstrip(", ")])

		# save to CSV
		saveCSV(topic_list, 'topic-list', folder = self.output_folder)
		saveCSV(topic_table, 'topic-table', folder = self.output_folder)

