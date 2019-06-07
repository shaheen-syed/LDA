# coding: utf-8

"""
	Created by:	Shaheen Syed
	Data:		August 2018

	The pre-processing phase can be seen as the process of going from a document source to an interpretable representation for the topic model algorithm. This phase is typically different 
	for full-text and abstract data. One of the main differences is that abstract data is often provided in a clean format, whereas full-text is commonly obtained by converting a PDF document 
	into its plain text representation.

	Within this phase, an important part is to filter out the content that is not important from a topic model's point-of-view, rather than from a human’s point-of-view. Abstract data 
	usually comes in a clean format of around 300--400 words, and little additional text is added to it; typically the copyright statement is the only text that should be removed. 
	In contrast, full-text articles can contain a lot of additional text that has been added by the publisher. This is article meta-data and boilerplate. It is important that such 
	additional text is removed, and various methods to do so exist. Examples include: deleting the first cover page; deleting the first n-bits of the content; using regular expressions 
	or other pattern matching techniques to find and remove additional text, or more advanced methods. For full-text articles, a choice can be made to also exclude the reference 
	list or acknowledgment section of the publication.

	Latent Dirichlet allocation, as well as other probabilistic topic models,  are bag-of-words (BOW) models. Therefore, the words within the documents need to be tokenized; the process 
	of obtaining individual words (also known as unigrams) from sentences. For English text, splitting words on white spaces would be the easiest example. Besides obtaining unigrams, it 
	is also important to find multi-word expressions, such as two-word (bi-grams) or multi-word (n-grams) combinations. Named entity recognition (NER)---a technique from natural 
	language processing (NLP)---can, for instance, be used to find multi-word expressions related to names, nationalities, companies, locations, and objects within the documents. 
	The inclusion of bi-grams and entities allows for a richer bag-of-words representation than a standard unigram representation. Documents from languages with implicit word boundaries 
	may require a more advanced type of tokenization.

	Although all tokens within a document serve an important grammatical or syntactical function, for topic modeling they are not all equally important. Words need to be filtered for 
	numbers, punctuation marks, and single-character words as they bear no topical meaning. Furthermore, stop words (e.g., the, is, a, which) are words that have no specific meaning 
	from a topical point-of-view, and such words need to be removed as well. For English, and a number of other languages, there exist fixed lists of stop words that can easily be used 
	(many NLP packages such as NLTK and Spacy include them). However, it is important to also create a domain-specific (also referred to as corpus-specific) list of stop words and filter 
	for those words. Such domain-specific stop words can also become apparent in the evaluation phase. If this is the case, going back to the pre-processing phase and excluding them would 
	be a good approach. Another approach to removing stop words is to use TF-IDF and include or exclude words within a certain threshold. Contrary to our analysis, some have indicated 
	that removing stop words have no substantial effect on model likelihood, topic coherence, or classification accuracy.

	For grammatical reasons, different word forms or derivationally related words can have a similar meaning and, ideally, for a topic model analysis, such terms need to be 
	grouped (i.e., they need to be normalized). Stemming and lemmatization are two NLP techniques to reduce inflectional and derivational forms of words to a common base 
	form. Stemming heuristically cuts off derivational affixes to achieve some normalization, albeit crude in most cases. Stemming loses the ability to relate stemmed words 
	back to their original part-of-speech, such as verbs or nouns, and decreases the interpretability of topics in later stages. Lemmatization is a more sophisticated normalization method 
	that uses a vocabulary and morphological analysis to reduce words to their base form, called lemma. For increased topic interpretability, we recommend lemmatization over stemming. 
	Additionally, uppercase and lowercase words can be grouped for further normalization. The process of normalization is particularly critical for languages with a richer 
	morphology. Failing to do can cause the vocabulary to be overly large, which can slow down posterior inference, and can lead to topics of poor quality.

	For reference articles see:
	Syed, S., Borit, M., & Spruit, M. (2018). Narrow lenses for capturing the complexity of fisheries: A topic analysis of fisheries science from 1990 to 2016. Fish and Fisheries, 19(4), 643–661. http://doi.org/10.1111/faf.12280
	Syed, S., & Spruit, M. (2017). Full-Text or Abstract? Examining Topic Coherence Scores Using Latent Dirichlet Allocation. In 2017 IEEE International Conference on Data Science and Advanced Analytics (DSAA) (pp. 165–174). Tokyo, Japan: IEEE. http://doi.org/10.1109/DSAA.2017.61
	Syed, S., & Spruit, M. (2018a). Exploring Symmetrical and Asymmetrical Dirichlet Priors for Latent Dirichlet Allocation. International Journal of Semantic Computing, 12(3), 399–423. http://doi.org/10.1142/S1793351X18400184
	Syed, S., & Spruit, M. (2018b). Selecting Priors for Latent Dirichlet Allocation. In 2018 IEEE 12th International Conference on Semantic Computing (ICSC) (pp. 194–202). Laguna Hills, CA, USA: IEEE. http://doi.org/10.1109/ICSC.2018.00035
	Syed, S., & Weber, C. T. (2018). Using Machine Learning to Uncover Latent Research Topics in Fishery Models. Reviews in Fisheries Science & Aquaculture, 26(3), 319–336. http://doi.org/10.1080/23308249.2017.1416331

"""

# packages and modules
import logging, sys, spacy
from collections import Counter
import itertools
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

	def full_text_preprocessing(self, pdf_folder = os.path.join('files', 'pdf')):


		"""
			preprocess full-text publications
			- convert pdf to plain text
			- correct for carriage returns
			- correct for end-of-line hyphenation
			- remove boilerplate
			- remove bibliography
			- remove acknowledgements

			Parameters
			----------
			pdf_folder : os.path
				location where PDF documents are stored
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

			Parameters
			----------
			min_bigram_count : int (optional)
				frequency of bigram to occur to include into list of bigrams. Thus lower frequency than min_bigram_count will not be included.
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

			