# -*- coding: utf-8 -*-

"""
	Created by:	Shaheen Syed
	Data: 		August 2018
"""

# packages and modules
import logging, os, requests, textract, glob2, sys, csv
from datetime import datetime
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora, models

def set_logger(folder_name = 'logs'):

	"""
		Set up the logging to console layout

		Parameters
		----------
		folder_name : string, optional
				name of the folder where the logs can be saved to

	"""

	# create the logging folder if not exists
	create_directory(folder_name)

	# define the name of the log file
	log_file_name = os.path.join(folder_name, '{:%Y%m%d%H%M%S}.log'.format(datetime.now()))

	# set up the logger layout to console
	logging.basicConfig(filename=log_file_name, level=logging.NOTSET)
	console = logging.StreamHandler()
	formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)
	logger = logging.getLogger(__name__)


def create_directory(name):

	"""
		Create directory if not exists

		Parameters
		----------
		name : string
				name of the folder to be created

	"""

	try:
		if not os.path.exists(name):
			os.makedirs(name)
			logging.info('Created directory: {}'.format(name))
	except Exception, e:
		logging.error('[createDirectory] : {}'.format(e))
		exit(1)


def get_HTTPHeaders():

	"""
		Create http header so a crawler will be identified as normal browser

		Returns
		--------
		http_header : dictionary
			html headers
	"""

	return {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit 537.36 (KHTML, like Gecko) Chrome", 
				"Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
				"Accept-Language": "en-US,en;q=0.5"}


def return_html(url):

	"""
		Scrape html content from url

		Parameters
		---------
		url : string
			http link to a website

		Returns
		-------
		html: request html object
			the full content of the html page
	"""

	try:
		# retrieve html content
		html = requests.get(url, headers=get_HTTPHeaders())
		# check for status
		if html.status_code == requests.codes.ok:
			return html
		else:
			logging.error("[return_html] invalid status code: {}".format(html.status_code))
			return None
	except Exception,e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		return None


def save_pdf(url, folder, name, overwrite = True):

	"""
		Save PDF file from the web to disk

		Parameters
		-----------
		url : string
			http link to PDF file
		folder : os.path
			location where to store the PDF file
		name : string
			name of the PDF file
		overwrite: Boolean (optional)
			if PDF already on disk, set to True if needs to be overwritten, or False to skip
	"""

	# create folder if not exists
	create_directory(folder)

	# check if file exists
	file_exists = os.path.exists(os.path.join(folder, name))

	# retrieve PDF from web
	if overwrite == True or file_exists == False:
		
		try:
			# retrieve pdf content
			response = requests.get(url, headers= get_HTTPHeaders(), stream=True)

			# save to folder
			with open('{}/{}'.format(folder, name), 'wb') as f:
				f.write(response.content)

		except Exception, e:
			logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
			exit(1)


def pdf_to_plain(pdf_file):

	
	"""
		Read PDF file and convert to plain text

		Parameters
		----------
		pdf_file : string
			location of pdf file

		Returns
		---------
		plain_pdf = string
			plain text version of the PDF file.
	"""


	try:

		# use textract to convert PDF to plain text
		return textract.process(pdf_file, encoding='utf8')

	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		return None
		

def read_directory(directory):

	"""

		Read file names from directory recursively

		Parameters
		----------
		directory : string
					directory/folder name where to read the file names from

		Returns
		---------
		files : list of strings
    			list of file names
	"""
	
	try:
		return glob2.glob(os.path.join( directory, '**' , '*.*'))
	except Exception, e:
		logging.error('[read_directory] : {}'.format(e))
		exit(1)



def word_tokenizer(text):

	"""
		Function to return individual words from text. Note that lemma of word is returned excluding numbers, stopwords and single character words
	"""

	# start tokenizing
	try:
		# # create spacey object
		# spacy_doc = nlp(text)
		# Lemmatize tokens, remove punctuation and remove stopwords.
		return  [token.lemma_ for token in text if token.is_alpha and not token.is_stop and len(token) > 1]
	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def get_bigrams(text):

	"""
		Get all the bigrams from a given text
	"""

	try:
		return list(nltk.bigrams(text.split()))
	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def named_entity_recognition(text):

	"""
		Perform named entity recognition on text to return all entities found that are at least two words
	"""

	try:
		# create spacey object
		ents = text.ents
		entities = [str(entity).lower() for entity in ents if len(str(entity).split()) > 2]
		return [ent.strip() for ent in entities if not any(char.isdigit() for char in ent) and all(ord(char) < 128 for char in ent)]
	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def get_dic_corpus(file_folder):

	"""
		Read dictionary and corpus for Gensim LDA

		Parameters
		-----------
		file_folder : os.path
			locatino of dictionary and corpus

		Returns
		dictionary : dict()
			LDA dictionary
		corpus : mm
			LDA corpus
	"""

	# create full path of dictionary
	dic_path = os.path.join(file_folder, 'dictionary.dict')
	# create full path of corpus
	corpus_path = os.path.join(file_folder, 'corpus.mm')


	# check if dictionary exists
	if os.path.exists(dic_path):
		dictionary = corpora.Dictionary.load(dic_path)
	else:
		logging.error('LDA dictionary not found')
		exit(1)

	# check if corpus exists
	if os.path.exists(corpus_path):
		corpus = corpora.MmCorpus(corpus_path)
	else:
		logging.error('LDA corpus not found')
		exit(1)

	return dictionary, corpus


def load_lda_model(model_location):

	"""
		Load the LDA model

		Parameters
		-----------
		model_location : os.path()
			location of LDA Model

		Returns
		-------
		model : gensim.models.LdaModel
			trained gensim lda model
	"""

	model_path = os.path.join(model_location, 'lda.model')

	if os.path.exists(model_path):
		return  models.LdaModel.load(model_path)
	else:
		logging.error('LDA model not found')
		exit(1)


def get_topic_label(k, labels_available = True):

	"""
		Return topic label

		Parameters
		-----------
		k : int
			topic id from lda model
		labels_available: Boolean (optional)
			if set to True, then labels are present, otherwise, return e.g. 'topic (1)' string. Default is true

		Returns
		-------
		label: string
			label for topic word distribution

	"""

	if not labels_available:

		return 'Topic {}'.format(k)

	else:

		topics = {	0 : 'Convergence',
					1 : 'State, Policy, Action',
					2 : 'Linear Algebra',
					3 : 'NLP',
					4 : 'Inference',
					5 : 'Computer Vision',
					6 : 'Graphical Models',
					7 : 'Neural Network Learning',
					8 : 'Stimulus Response',
					9 : 'Neural Network Structure'}
		
		return topics[k]


def save_csv(data, name, folder):

	"""
		Save list of list as CSV (comma separated values)

		Parameters
		----------
		data : list of list
    			A list of lists that contain data to be stored into a CSV file format
    	name : string
    			The name of the file you want to give it
    	folder: string
    			The folder location
	"""
	
	try:

		# create folder name as directory if not exists
		create_directory(folder)

		# create the path name (allows for .csv and no .csv extension to be handled correctly)
		suffix = '.csv'
		if name[-4:] != suffix:
			name += suffix

		# create the file name
		path = os.path.join(folder, name)

		# save data to folder with name
		with open(path, "w") as f:
			writer = csv.writer(f, lineterminator='\n')
			writer.writerows(data)

	except Exception, e:

		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def read_csv(filename, folder = None):

	"""
		Read CSV file and return as a list

		Parameters
		---------
		filename : string
			name of the csv file
		folder : string (optional)
			name of the folder where the csv file can be read

		Returns
		--------

	"""

	if folder is not None:
		filename = os.path.join(folder, filename)
	
	try:
		# increate CSV max size
		csv.field_size_limit(sys.maxsize)
		
		# open the filename
		with open(filename, 'rb') as f:
			# create the reader
			reader = csv.reader(f)
			# return csv as list
			return list(reader)
	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)