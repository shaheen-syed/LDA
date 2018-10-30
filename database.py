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
from pymongo import MongoClient
import time, logging, sys
from bson.objectid import ObjectId


class MongoDatabase:

	def __init__(self, client = 'LDA-github'):

		self.client = MongoClient()
		self.db = self.client[client]


		# set utf8 encoding
		reload(sys)
		sys.setdefaultencoding('utf8')


	def read_collection(self, collection):

		"""
			Read all documents in a certain collection
		"""

		try:
			return self.db[collection].find({}, no_cursor_timeout=True)
		except Exception, e:
			logging.error("[{}] : {}".format(sys._getframe().f_code.co_name,e))
			exit(1)

	def insert_one_to_collection(self, collection, doc):


		"""
			Insert one document to a collection
		"""

		try:
			self.db[collection].insert_one(doc)
		except Exception, e:
			logging.error("[{}] : {}".format(sys._getframe().f_code.co_name,e))
			exit(1)


	def update_collection(self, collection, doc):


		"""
			Update document to a collection
		"""

		try:	
			self.db[collection].update({'_id' : ObjectId(doc['_id'])},
									doc
									,upsert = False)
		except Exception, e:
			logging.error("[{}] : {}".format(sys._getframe().f_code.co_name,e))
			exit(1)
