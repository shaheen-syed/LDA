# -*- coding: utf-8 -*-

"""
	IMPORT MODULES
"""
import logging, sys, re
from gensim import corpora, models

"""
	IMPORT CLASSES
"""
from helper_functions import *



class Datamining():

	def __init__(self):

		logging.info('Initialized {}'.format(self.__class__.__name__))

		# set utf8 encoding
		reload(sys)
		sys.setdefaultencoding('utf8')


	def execute_lda(self, file_folder, save_folder):

		"""
			Perform LDA inference with gensim + grid search approach
		"""

		logging.info('Start {}'.format(sys._getframe().f_code.co_name))

		"""
			Define variables for LDA inference
			- number of topics
			- dirichlet priors 'auto', 'symmetric', 'asymmetric'
			- random initialization
			- passes over the full corpus
			- max iterations for convergence
		"""

		# number of topics
		K = range(2,20 + 1)
		# different dirichlet priors
		dir_priors = ['auto']
		# random initiatilizations
		random_states = [42,99]
		# passes over the corpus
		num_passes = [5,10,15,20]
		# max iteration for convergence
		iterations = [200]

		# read dictionary and corpus
		dictionary, corpus = get_dic_corpus(file_folder)

		# start grid search
		for k in K:

			logging.debug('topics: {}'.format(k))

			for dir_prior in dir_priors:

				logging.debug('Dirichlet prior: {}'.format(dir_prior))

				for random_state in random_states:

					logging.debug('Random state: {}'.format(random_state))

					for num_pass in num_passes:

						logging.debug('Number of passes: {}'.format(num_pass))

						for iteration in iterations:

							logging.debug('Iterations: {}'.format(iteration))

							# check if model already created
							target_folder = '{}/{}/{}/{}/{}/{}'.format(save_folder, k, dir_prior, random_state, num_pass, iteration)

							if not (os.path.exists(target_folder)):

								# create target folder to save LDA model to
								create_dir(target_folder)

								model = models.LdaModel(corpus, 
														id2word = dictionary,
														num_topics = k,
														iterations= iteration, 
														passes = num_pass, 
														minimum_probability = 0, 
														alpha = dir_prior, 
														eta = dir_prior, 
														eval_every = None,
														random_state= random_state)

								# save LDA model
								model.save('{}/lda.model'.format(target_folder))
								
							else:
								logging.info('LDA model already exists, skipping ...')
