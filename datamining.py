# -*- coding: utf-8 -*-

"""
	Created by:		Shaheen Syed
	Date:			August 2018

	The data mining phase involves fitting or training the LDA model. It also involves a careful analysis of the hyper-parameters and the creation of different LDA models. Similarly to 
	the transformation phase, the use of a certain LDA module or software tool determines what parameters and hyper-parameters can be adjusted.

	Since calculating a closed form solution of the LDA model is intractable, approximate posterior inference is used to create the distributions of words in topics, and topics in 
	documents. To avoid local minima, both in the case of variational and sampling-based inference techniques, the initialization of the model is an important consideration in the 
	data mining phase. Thus, regardless the initialization, and regardless the inference method used, to guard for problems of local minima, multiple starting points should be used 
	to improve the stability of the inferred latent variables.

	Running the inference is the most important step in the data mining phase. It results in the discovery of the latent variables (words in topics, and topics in documents). 
	The convergence of the model (e.g., the likelihood) should be closely monitored. The time typically depends on initialization, the number of documents, model complexity, 
	and inference technique. A straightforward approach to optimize for the various hyper-parameters would be to perform a grid-search and inferring LDA models for combinations of 
	them. Such hyper-parameters include the number of epochs or passes over the corpus, the number of iterations for convergence, the number of topics, the types of Dirichlet priors, 
	and the starting points.

	For reference articles see:
	Syed, S., Borit, M., & Spruit, M. (2018). Narrow lenses for capturing the complexity of fisheries: A topic analysis of fisheries science from 1990 to 2016. Fish and Fisheries, 19(4), 643–661. http://doi.org/10.1111/faf.12280
	Syed, S., & Spruit, M. (2017). Full-Text or Abstract? Examining Topic Coherence Scores Using Latent Dirichlet Allocation. In 2017 IEEE International Conference on Data Science and Advanced Analytics (DSAA) (pp. 165–174). Tokyo, Japan: IEEE. http://doi.org/10.1109/DSAA.2017.61
	Syed, S., & Spruit, M. (2018a). Exploring Symmetrical and Asymmetrical Dirichlet Priors for Latent Dirichlet Allocation. International Journal of Semantic Computing, 12(3), 399–423. http://doi.org/10.1142/S1793351X18400184
	Syed, S., & Spruit, M. (2018b). Selecting Priors for Latent Dirichlet Allocation. In 2018 IEEE 12th International Conference on Semantic Computing (ICSC) (pp. 194–202). Laguna Hills, CA, USA: IEEE. http://doi.org/10.1109/ICSC.2018.00035
	Syed, S., & Weber, C. T. (2018). Using Machine Learning to Uncover Latent Research Topics in Fishery Models. Reviews in Fisheries Science & Aquaculture, 26(3), 319–336. http://doi.org/10.1080/23308249.2017.1416331


"""

# packages and modules
import logging, sys, re
from gensim import corpora, models
from helper_functions import *


class Datamining():

	def __init__(self):

		logging.info('Initialized {}'.format(self.__class__.__name__))

	def execute_lda(self, file_folder = os.path.join('files', 'lda'), save_folder = os.path.join('files', 'models')):

		"""
			Perform LDA inference with gensim + grid search approach

			Parameters
			----------
			file_folder: os.path
				location of the dictionary and corpus for gensim
			save_folder: os.path
				location to save the lda models to
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
							target_folder = os.path.join(save_folder, str(k), dir_prior, str(random_state), str(num_pass), str(iteration))

							if not (os.path.exists(target_folder)):

								# create target folder to save LDA model to
								create_directory(target_folder)

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
								model.save(os.path.join(target_folder, 'lda.model'))
								
							else:
								logging.info('LDA model already exists, skipping ...')
