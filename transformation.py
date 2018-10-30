"""
	IMPORT MODULES
"""
import logging, sys, re
from gensim import corpora

"""
	IMPORT CLASSES
"""
from database import MongoDatabase
from helper_functions import *



class Transformation():

	def __init__(self):

		logging.info('Initialized {}'.format(self.__class__.__name__))

		# instantiate database
		self.db = MongoDatabase()

		# set utf8 encoding
		reload(sys)
		sys.setdefaultencoding('utf8')


	def transform_for_lda(self, save_folder, min_num_word, max_percentile_words):

		"""
			Transform the corpus of words into LDA input featues
			This transformation is typically dictated by the LDA tool you use
			here we use Gensim which require a dictionary and corpus in a specific format
		"""

		# read document collection
		D = self.db.read_collection(collection = 'publications_raw')

		# create word input features per document
		texts = [x['tokens'] for x in D]

		# create dictionary of docs and filter away to % and bottom frequency
		dictionary = corpora.Dictionary(texts)
		dictionary.filter_extremes(no_below = min_num_word, no_above = max_percentile_words)

		# create save folder if not exists
		create_dir(save_folder)

		# store the dictionary, for future reference
		dictionary.save('{}/dictionary.dict'.format(save_folder))
		
		# create vector based corpus => bag of words with frequencies stored as a sparsed vector
		corpus = [dictionary.doc2bow(text) for text in texts]
		
		# store to disk, for later use
		corpora.MmCorpus.serialize('{}/corpus.mm'.format(save_folder), corpus)

