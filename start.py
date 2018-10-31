# -*- coding: utf-8 -*-

"""
	Created by Shaheen Syed

	For reference articles see:
	Syed, S., Borit, M., & Spruit, M. (2018). Narrow lenses for capturing the complexity of fisheries: A topic analysis of fisheries science from 1990 to 2016. Fish and Fisheries, 19(4), 643–661. http://doi.org/10.1111/faf.12280
	Syed, S., & Spruit, M. (2017). Full-Text or Abstract? Examining Topic Coherence Scores Using Latent Dirichlet Allocation. In 2017 IEEE International Conference on Data Science and Advanced Analytics (DSAA) (pp. 165–174). Tokyo, Japan: IEEE. http://doi.org/10.1109/DSAA.2017.61
	Syed, S., & Spruit, M. (2018a). Exploring Symmetrical and Asymmetrical Dirichlet Priors for Latent Dirichlet Allocation. International Journal of Semantic Computing, 12(3), 399–423. http://doi.org/10.1142/S1793351X18400184
	Syed, S., & Spruit, M. (2018b). Selecting Priors for Latent Dirichlet Allocation. In 2018 IEEE 12th International Conference on Semantic Computing (ICSC) (pp. 194–202). Laguna Hills, CA, USA: IEEE. http://doi.org/10.1109/ICSC.2018.00035
	Syed, S., & Weber, C. T. (2018). Using Machine Learning to Uncover Latent Research Topics in Fishery Models. Reviews in Fisheries Science & Aquaculture, 26(3), 319–336. http://doi.org/10.1080/23308249.2017.1416331

	PACKAGES NEEDED TO INSTALL: 
	logging, pymongo, requests, textract, glob2, spacy, nltk, seaborn

	pip install -U spacy
	python -m spacy download en

	use chardet==2.3.0 for textract (mainly convert pdf to plain text). The new version does not work properly in some cases where encoding
	cannot be retrieved
"""


"""
	IMPORT CLASSES
"""
from extraction import Extraction
from preprocessing import Preprocessing
from transformation import Transformation
from datamining import Datamining
from evaluation import Evaluation
from interpretation import Interpretation
from helper_functions import *


"""
	IMPORT MODULES
"""
import logging
from datetime import datetime


"""
	SWITCHES
"""
EXTRACTION = False
PREPROCESSING = False
TRANSFORMATION = False
DATAMINING = False
EVALUATION = False
INTERPRETATION = False


if __name__ == "__main__":

	# create logging to console
	set_logger()

	# perform extraction of full-text articles
	if EXTRACTION:

		# instantiate Extraction class
		extraction = Extraction()

		# extract publications from NIPS
		extraction.extract_publications()


	if PREPROCESSING:

		"""
			Preprocess the publications
		"""

		# instantiate Preprocessing class
		preprocessing = Preprocessing()

		# pre-process full-text articles
		preprocessing.full_text_preprocessing(pdf_folder = '../PDF')

		# preprocessing general
		preprocessing.general_preprocessing()


	if TRANSFORMATION:
		
		"""
			Transform data into Gensim dictionary, corpus and BOW features
		"""

		# instantiate Transformation class
		transformation = Transformation()

		# transform data to make it suitable for LDA analysis
		transformation.transform_for_lda(save_folder = 'files', min_num_word = 5, max_percentile_words = 0.90)


	if DATAMINING:

		"""
			Perform the LDA analysis
		"""

		# instantiate Datamining class
		datamining = Datamining()

		# perform LDA inference with grid search
		datamining.execute_lda(file_folder = 'files', save_folder = 'models')


	if EVALUATION:

		"""
			Evaluate models and obtain the best model.
		"""

		# instantiate evaluation class
		evaluation = Evaluation()

		# calculate coherence score for each created model
		evaluation.calculate_coherence(file_folder = 'files', models_folder = 'models')

		# plot coherence scores
		evaluation.plot_coherence()

		# output LDA words and probabilities
		evaluation.output_lda_topics()


	if INTERPRETATION:

		"""
			Interpret the LDA model
		"""

		# instantiate interpretation class
		interpretation = Interpretation()

		# infer the documen topic distribution per publication
		interpretation.infer_document_topic_distribution()

		# obtain list of document titles per topic
		interpretation.get_document_title_per_topic()

		# plot topics over time
		interpretation.plot_topics_over_time()

		# plot topics over time stacked
		interpretation.plot_topics_over_time_stacked()

		# plot topic co-occurrence
		interpretation.plot_topic_co_occurrence()
