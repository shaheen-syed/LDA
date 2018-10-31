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
	IMPORTS
"""
import logging, os, requests, textract, glob2, sys, csv
from datetime import datetime
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora, models

"""
	Helper functions for various generic tasks
"""

def set_logger():

	"""
		Set up the logging to console
	"""

	create_dir('logs')
	logging.basicConfig(filename='logs/' + '{:%Y%m%d%H%M%S}'.format(datetime.now()) + '.log' ,level=logging.NOTSET)
	console = logging.StreamHandler()
	formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)
	logger = logging.getLogger(__name__)


def create_dir(name):

	"""
		Create directory if not exists
	"""

	try:
		if not os.path.exists(name):
			os.makedirs(name)
			logging.debug('Created directory: {}'.format(name))
	except Exception, e:
		logging.error('[createDirectory] : {}'.format(e))


def get_HTTPHeaders():

	"""
		Create http header so a crawler will be identified as normal browser
	"""

	return {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit 537.36 (KHTML, like Gecko) Chrome", 
				"Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
				"Accept-Language": "en-US,en;q=0.5"}


def return_html(url):

	"""
		Scrape html content from url
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
		logging.error("[return_html] error retrieving html content: {}".format(str(e)))
		return None

def save_pdf(url, folder, overwrite, name):

	"""
		Save PDF file from the web to disk
	"""

	# create folder if not exists
	create_dir(folder)

	# check if file exists
	file_exists = os.path.exists('{}/{}'.format(folder, name))

	# retrieve PDF from web
	if overwrite == True or file_exists == False:
		try:
			# retrieve pdf content
			response = requests.get(url, headers=get_HTTPHeaders(), stream=True)

			# save to folder
			with open('{}/{}'.format(folder, name), 'wb') as f:
				f.write(response.content)

		except Exception, e:
			logging.error('[save_pdf] : {}'.format(e))

def pdf_to_plain(pdf_file):

	
	"""
		Read PDF file and convert to plain text
	"""


	try:
		return textract.process(pdf_file, encoding='utf8')
	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		return None

def read_directory(directory):

	"""
		Read files from a directory
	"""
	
	try:
		return glob2.glob('{}/**/*.*'.format(directory))
	except Exception, e:
		logging.error('[read_directory] : {}'.format(e))
		return None


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
	"""


	# check if dictionary exists
	if (os.path.exists('{}/dictionary.dict'.format(file_folder))):
		dictionary = corpora.Dictionary.load('{}/dictionary.dict'.format(file_folder))
	else:
		logging.error('LDA dictionary not found')
		exit(1)

	# check if corpus exists
	if (os.path.exists('{}/corpus.mm'.format(file_folder))):
		corpus = corpora.MmCorpus('{}/corpus.mm'.format(file_folder))
	else:
		logging.error('LDA corpus not found')
		exit(1)

	return dictionary, corpus


def load_LDA_model(model_location):

	"""
		Load the LDA model
	"""

	if os.path.exists(model_location):
		return  models.LdaModel.load(model_location)
	else:
		logging.error('LDA Model not found')
		exit(1)
	

def get_topic_label(k):

	"""
		Obtain the label for a topic
		Note that LDA topics start from 0
	"""

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


def saveCSV(data, name, folder):

	"""
		Save list as CSV file
	"""
	
	with open(folder + '/' + name + '.csv', "w") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(data)