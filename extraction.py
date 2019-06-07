# -*- coding: utf-8 -*-

"""
	Created by: 	Shaheen Syed
	Date:			August 2018

	The extraction phase involves the process of obtaining a set of documents from a repository, such as Scopus or the Web of Science, or it can involve the steps of scraping a 
	publisher’s website to retrieve full-text articles (typically in PDF format). Scopus generally provides publication abstracts, including all the meta-data (journal, authors, 
	affiliations, publication date) through various APIs. The upside of using an API is that publication content is easily obtained for a large number of documents simultaneously, 
	however, these APIs often do not provide full-text for all the publications or journals of interest. In these cases, scraping a publisher’s websites can be an alternative solution. 
	This process involves building many handcrafted crawlers, as each publisher lists their publications in a different manner on their website. Download limits should always be 
	respected when building such scripts. Another option would be to manually download articles, although such approaches might not be feasible if the document collection of interest 
	contains thousands or tens of thousands of articles. To enable a comparison of topics by time, or a comparison of topics by journals, it is important to store this information 
	alongside the content of the document.



	For reference articles see:
	Syed, S., Borit, M., & Spruit, M. (2018). Narrow lenses for capturing the complexity of fisheries: A topic analysis of fisheries science from 1990 to 2016. Fish and Fisheries, 19(4), 643–661. http://doi.org/10.1111/faf.12280
	Syed, S., & Spruit, M. (2017). Full-Text or Abstract? Examining Topic Coherence Scores Using Latent Dirichlet Allocation. In 2017 IEEE International Conference on Data Science and Advanced Analytics (DSAA) (pp. 165–174). Tokyo, Japan: IEEE. http://doi.org/10.1109/DSAA.2017.61
	Syed, S., & Spruit, M. (2018a). Exploring Symmetrical and Asymmetrical Dirichlet Priors for Latent Dirichlet Allocation. International Journal of Semantic Computing, 12(3), 399–423. http://doi.org/10.1142/S1793351X18400184
	Syed, S., & Spruit, M. (2018b). Selecting Priors for Latent Dirichlet Allocation. In 2018 IEEE 12th International Conference on Semantic Computing (ICSC) (pp. 194–202). Laguna Hills, CA, USA: IEEE. http://doi.org/10.1109/ICSC.2018.00035
	Syed, S., & Weber, C. T. (2018). Using Machine Learning to Uncover Latent Research Topics in Fishery Models. Reviews in Fisheries Science & Aquaculture, 26(3), 319–336. http://doi.org/10.1080/23308249.2017.1416331


	The code below downloads PDFs (full-text) publications from the NIPS proceedings. Note that the code only works for the structure of the NIPS website, sinch each publisher will structure their html pages
	differently. A careful analysis of the underlying html structure should be done before understanding how download PDF links can be obtained.
"""

# modules and packages
import logging, sys, re
from helper_functions import *


class Extraction():

	def __init__(self):

		logging.info('Initialized {}'.format(self.__class__.__name__))		

	def extract_publications(self, save_folder = os.path.join('files', 'pdf')):

		"""
			Crawl the website and extract links to publications and download full-text pdf
			This is only one example that works for the NIPS site. Other websites will need different
			crawling techniques

			Parameters
			----------
			save_folder : os.path
				folder location where PDFs need to be saved
		"""

		# crawl the first page to retrieve links to next page
		journal = 'NIPS'
		domain = 'https://papers.nips.cc'
		p1_content = return_html(domain).text
		p1_links = re.findall(r'<a href="(/book.*?)">', p1_content) 

		# go to second page
		for l in p1_links:

			# extract year from link
			year = l[-4:]

			# construct link to second page
			l2 = '{}{}'.format(domain, l)

			# scrape content of second page
			p2_content = return_html(l2).text

			# retrieve all links of second page
			p2_links = re.findall(r'<a href="(/paper.*?)">', p2_content)

			# read publications from file so we don't process them again
			processed_publications = [x.split(os.sep)[-1][0:-4] for x in read_directory(os.path.join(save_folder, journal, year))]
			

			# extract links to third page, which is where the paper overviews are
			for i, l2 in enumerate(p2_links):

				logging.debug('Year: {} , Publication {}/{}'.format(year, i + 1, len(p2_links)))

				# check if publication already processed
				if l2.split(os.sep)[-1] in processed_publications:
					logging.debug('publication already present, skipping ...')
					continue

				# construct link to third page
				l3 = '{}{}'.format(domain, l2)

				# scrape content of the third page
				p3_content = return_html(l3).text

				# extract PDF link from page and download PDF
				try:
					pdf_link = '{}{}'.format(domain, re.findall(r'href="(/paper/.*\.pdf)">', p3_content)[0])
					pdf_name = pdf_link.split('/')[-1:][0]

					# download pdf
					save_pdf(pdf_link, 
								folder = os.path.join(save_folder, journal, year), 
								name = pdf_name,
								overwrite = False)
				except Exception, e:
					logging.warning('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
					continue


