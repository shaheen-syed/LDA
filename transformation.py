# -*- coding: utf-8 -*-

"""
	Created by:		Shaheen Syed
	Date:			August 2018
	
	The transformation phase includes the creation of a dictionary of words and preparing the data for the topic model software or package. The dictionary of words (also referred 
	to as the vocabulary) is typically a list of unique words represented as integers. For example, ‘fish’ is 1, ‘population’ is 2 and so on. The length of the dictionary is thus 
	the number of unique words within the corpus; normalization reduces the length of the dictionary, and speeds up the inference time. The next step is to represent the documents as 
	bag-of-words features. Doing this for all the documents creates a matrix (i.e., table) with the rows being the individual documents, the columns being the words within the 
	dictionary, and the cells being the frequency of that word within the document. This is one representation of bag-of-words features and other less sparse representations exist as 
	well. If not performed during the pre-processing phase, words that occur only once, and words that occur in roughly 90% of the documents can be eliminated as they serve no 
	discriminative topical significance. Especially omitting high frequently occurring words prevents such words from dominating all topics. Removing high and low-frequency words 
	(i.e., pruning) within a matrix representation is generally much faster to perform.

	Several LDA tools are available and each of them requires a slightly different transformation step to make the data suitable for topic analysis. However, in essence, 
	they require a conversion from words to bag-of-words representation, to some matrix representation of the full corpus. Several LDA packages exist that might be worth 
	exploring: Gensim, Mallet, Stanford Topic Modeling Toolbox, Yahoo! LDA, and Mr.~LDA.

	For reference articles see:
	Syed, S., Borit, M., & Spruit, M. (2018). Narrow lenses for capturing the complexity of fisheries: A topic analysis of fisheries science from 1990 to 2016. Fish and Fisheries, 19(4), 643–661. http://doi.org/10.1111/faf.12280
	Syed, S., & Spruit, M. (2017). Full-Text or Abstract? Examining Topic Coherence Scores Using Latent Dirichlet Allocation. In 2017 IEEE International Conference on Data Science and Advanced Analytics (DSAA) (pp. 165–174). Tokyo, Japan: IEEE. http://doi.org/10.1109/DSAA.2017.61
	Syed, S., & Spruit, M. (2018a). Exploring Symmetrical and Asymmetrical Dirichlet Priors for Latent Dirichlet Allocation. International Journal of Semantic Computing, 12(3), 399–423. http://doi.org/10.1142/S1793351X18400184
	Syed, S., & Spruit, M. (2018b). Selecting Priors for Latent Dirichlet Allocation. In 2018 IEEE 12th International Conference on Semantic Computing (ICSC) (pp. 194–202). Laguna Hills, CA, USA: IEEE. http://doi.org/10.1109/ICSC.2018.00035
	Syed, S., & Weber, C. T. (2018). Using Machine Learning to Uncover Latent Research Topics in Fishery Models. Reviews in Fisheries Science & Aquaculture, 26(3), 319–336. http://doi.org/10.1080/23308249.2017.1416331

"""

# packages and modules
import logging, sys, re
from gensim import corpora
from database import MongoDatabase
from helper_functions import *


class Transformation():

	def __init__(self):

		logging.info('Initialized {}'.format(self.__class__.__name__))

		# instantiate database
		self.db = MongoDatabase()

	def transform_for_lda(self, save_folder = os.path.join('files', 'lda'), no_below = 5, no_above = 0.90):

		"""
			Transform the corpus of words into LDA input features
			This transformation is typically dictated by the LDA tool you use
			here we use Gensim which require a dictionary and corpus in a specific format

			Parameters
			----------
			save_folder: os.path
				location to save the dictionary and corpus to
			no_below:	(int, optional)
				Keep tokens which are contained in at least no_below documents.
			no_above: (float, optional)
				Keep tokens which are contained in no more than no_above documents (fraction of total corpus size, not an absolute number).

		"""

		# read document collection
		D = self.db.read_collection(collection = 'publications_raw')

		# create word input features per document
		texts = [x['tokens'] for x in D]

		# create dictionary of docs and filter away to % and bottom frequency
		dictionary = corpora.Dictionary(texts)
		dictionary.filter_extremes(no_below = no_below, no_above = no_above)

		# create save folder if not exists
		create_directory(save_folder)

		# store the dictionary, for future reference
		dictionary.save(os.path.join(save_folder, 'dictionary.dict'))
		
		# create vector based corpus => bag of words with frequencies stored as a sparsed vector
		corpus = [dictionary.doc2bow(text) for text in texts]
		
		# store to disk, for later use
		corpora.MmCorpus.serialize(os.path.join(save_folder, 'corpus.mm'), corpus)
