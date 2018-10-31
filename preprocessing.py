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
import logging, sys, spacy
from collections import Counter
import itertools

"""
	IMPORT CLASSES
"""
from database import MongoDatabase
from helper_functions import *


class Preprocessing():

	def __init__(self):

		logging.info('Initialized {}'.format(self.__class__.__name__))

		# instantiate database
		self.db = MongoDatabase()

		# set utf8 encoding
		reload(sys)
		sys.setdefaultencoding('utf8')


	def full_text_preprocessing(self, pdf_folder):


		"""
			preprocess full-text publications
			- convert pdf to plain text
			- correct for carriage returns
			- correct for end-of-line hyphenation
			- remove boilerplate
			- remove bibliography
			- remove acknowledgements
		"""
		
		logging.info('Start {}'.format(sys._getframe().f_code.co_name))

		# read pdf files that need to be converted
		F = [x for x in read_directory(pdf_folder) if x[-4:] == '.pdf']

		# read documents from DB that have already been processed so we can skip them
		processed_documents = [ '{}-{}-{}'.format(x['journal'], x['year'], x['title']) for x in self.db.read_collection(collection = 'publications_raw')]

		# loop over each file and convert pdf to plain and save meta data to DB
		for i, f in enumerate(F):

			# extract meta data from folder structure and file name
			journal = f.split('/')[2]
			year = f.split('/')[3]
			title = f.split('/')[4].replace('-', ' ')[4:-4].strip()

			# console output
			print_doc_verbose(i, len(F), journal, year, title)

			# check if PDF has already been processed
			if '{}-{}-{}'.format(journal, year, title) in processed_documents:
				logging.info('PDF document already processed, skipping ...')
				continue

			# convert content of PDF to plain text
			content = pdf_to_plain(f)

			# check if content could be extracted
			if content is not None:
				
				# fix soft hyphen
				content = content.replace(u'\xad', "-")
				# fix em-dash
				content = content.replace(u'\u2014', "-")
				# fix en-dash
				content = content.replace(u'\u2013', "-")
				# minus sign
				content = content.replace(u'\u2212', "-")
				# fix hyphenation that occur just before a new line
				content = content.replace('-\n','')
				# remove new lines/carriage returns
				content = content.replace('\n',' ')

				# correct for ligatures
				content = content.replace(u'\ufb02', "fl")	# fl ligature
				content = content.replace(u'\ufb01', "fi")	# fi ligature
				content = content.replace(u'\ufb00', "ff")	# ff ligature
				content = content.replace(u'\ufb03', "ffi") # ffi ligature
				content = content.replace(u'\ufb04', "ffl") # ffl ligature

				""" 
					Remove boilerplate content:

					Especially journal publications have lots of boilerplate content on the titlepage. Removing of this is specific for each
					journal and you can use some regular expressions to identify and remove it.
				"""

				"""
					Remove acknowledgemends and/or references
					This is a somewhat crude example
				"""
				if content.rfind("References") > 0:
					content = content[:content.rfind("References")]
			
				"""
				 	Remove acknowledgements
				"""
				if content.rfind("Acknowledgment") > 0:
					content = content[:content.rfind("Acknowledgment")]
			
				# prepare dictionary to save into MongoDB
				doc = {	'journal' : journal, 'title' : title, 'year' : year, 'content' : content}

				# save to database
				self.db.insert_one_to_collection(doc = doc, collection = 'publications_raw')

	def general_preprocessing(self, min_bigram_count = 5):

		"""
			General preprocessing of publications (used for abstracts and full-text)
		"""

		logging.info('Start {}'.format(sys._getframe().f_code.co_name))

		# read document collection
		D = self.db.read_collection(collection = 'publications_raw')

		# setup spacy natural language processing object
		nlp = setup_spacy()

		# loop through the documents and correct content
		for i, d in enumerate(D):

			# check if tokens are already present, if so, skip
			if d.get('tokens') is None:

				# print to console
				print_doc_verbose(i, D.count(), d['journal'], d['year'], d['title'])

				# get content from document and convert to spacy object
				content = nlp(d['content'])

				# tokenize, lemmatization, remove punctuation, remove single character words
				unigrams = word_tokenizer(content)

				# get entities
				entities = named_entity_recognition(content)

				# get bigrams
				bigrams = get_bigrams(" ".join(unigrams))
				bigrams = [['{} {}'.format(x[0],x[1])] * y for x, y in Counter(bigrams).most_common() if y >= min_bigram_count]
				bigrams = list(itertools.chain(*bigrams))

				d['tokens'] = unigrams + bigrams + entities

				# save dictionary to datbase
				self.db.update_collection(collection = 'publications_raw', doc = d)

				
			else:
				logging.debug('Document already tokenized, skipping ...')



""" 

internal helper function

"""

def print_doc_verbose(i, total, journal, year, title):

	# console output
	logging.debug('processing file: {}/{}'.format(i+1,total))
	logging.debug('journal : {}'.format(journal))
	logging.debug('year : {}'.format(year))
	logging.debug('title : {}'.format(title))

def setup_spacy():

	# setting up spacy
	nlp = spacy.load('en')

	# add some more stopwords; apparently spacy does not contain all the stopwords
	for word in set(stopwords.words('english')):

		nlp.Defaults.stop_words.add(unicode(word))
		nlp.Defaults.stop_words.add(unicode(word.title()))

	for word in nlp.Defaults.stop_words:
		lex = nlp.vocab[word]
		lex.is_stop = True
	
	return nlp

			