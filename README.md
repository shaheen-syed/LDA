# LDA Document Similarity

This repository contains all the code to perform a topic model analysis with latent Dirichlet allocation on a set of documents. It follows a systematic process that starts with data collections to extracting useful knowledge from derives patterns, here latent topics. The systematic process is called Knowledge Discovery in Database and each step in the KDD process has its accompanying file name. The start.py file can be configured to turn on/off certain parts of the data workflow. The full workflow is shown below:

![ScreenShot](/workflow/lda-workflow.png)

## What to do first


Packages required:
requests, textract, glob2, csv, datetime, spacy, nltk, gensim, itemgetter, matplotlib, seaborn, pandas, numpy, pymongo, collections, itertools, re, logging, os, sys

Install spacy with the following commands:
```
pip install -U spacy
python -m spacy download en
```

install stopwords from nltk:
```
import nltk
nltk.download('stopwords')
```

important when converting pdf to plain text:
use chardet==2.3.0 for textract (to convert pdf to plain text). The latest version of chardet does not work properly in some cases where encoding cannot be retrieved

## How to run

```
python start.py
```

## Which switches to turn on (set to True)

Within start.py, there are 6 switches that can be set to True, either all at the same time, or by turning them on one-by-one

* EXTRACTION 
* PREPROCESSING
* TRANSFORMATION
* DATAMINING
* EVALUATION
* INTERPRETATION

They will be explained below

## step 1 - EXTRACTION

The extraction phase involves the process of obtaining a set of documents from a repository, such as Scopus or the Web of Science, or it can involve the steps of scraping a 
publisher’s website to retrieve full-text articles (typically in PDF format). Scopus generally provides publication abstracts, including all the meta-data (journal, authors, 
affiliations, publication date) through various APIs. The upside of using an API is that publication content is easily obtained for a large number of documents simultaneously, 
however, these APIs often do not provide full-text for all the publications or journals of interest. In these cases, scraping a publisher’s websites can be an alternative solution. 
This process involves building many handcrafted crawlers, as each publisher lists their publications in a different manner on their website. Download limits should always be 
respected when building such scripts. Another option would be to manually download articles, although such approaches might not be feasible if the document collection of interest 
contains thousands or tens of thousands of articles. To enable a comparison of topics by time, or a comparison of topics by journals, it is important to store this information 
alongside the content of the document.

## step 2 - PREPROCESSING

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

## step 3 - TRANSFORMATION

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
exploring: Gensim, Mallet, Stanford Topic Modeling Toolbox, Yahoo! LDA, and Mr. LDA.

## step 4 - DATAMINING

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


## step 5 - EVALUATION

The evaluation phase includes a careful analysis and inspection of the latent variables from the various created LDA models. Since LDA is an unsupervised machine learning technique, 
extra care should be given during this post-analysis phase; in contrast to, for example, supervised methods where typically a labeled gold-standard dataset exist. 

Measures such as predictive likelihood on held-out data have been proposed to evaluate the quality of generated topics. However, such measures correlate negatively with human 
interpretability, making topics with high predictive likelihood less coherent from a human perspective. High-quality or coherent latent topics are of particular importance when 
they are used to browse document collections or understand the trends and development within a particular research field. As a result, researchers have proposed topic coherence measures, 
which are a quantitative approach to automatically uncover the coherence of topics. Topics are considered to be coherent if all or most of the words (e.g., a topic's top-N words) are 
related. Topic coherence measures aim to find measures that correlate highly with human topic evaluation, such as topic ranking data obtained by, for example, word and topic intrusion 
tests. Human topic ranking data are often considered the gold standard and, consequently, a measure that correlates well is a good indicator for topic interpretability. 

Exploring the topics by a human evaluator is considered the best approach. However, since this involves inspecting all the different models, this approach might not be feasible. 
Topic coherence measures can quantitatively calculate a proxy for topic quality, and per our analysis, topics with high coherence were considered interpretable by domain experts. 
Combing coherence measures with a manual inspection is thus a good approach to find the LDA model that result in meaningful and interpretable topics. In short, three questions 
should be answered satisfactory: 

*	Are topics meaningful, interpretable, coherent and useful?
*	Are topics within documents meaningful, appropriate and useful?  
*	Do the topics facilitate a better understanding of the underlying corpus?

The evaluation phase can also result in topics that are very similar (i.e., identical topics), topics that should ideally be merged or split (i.e., chained or mixed topics), topics 
that are un-interpretable (i.e. nonsensical), or topics that contain unimportant, too specific, or too general words. In those cases, it would be wise to revisit the pre-processing 
phase and repeat the analysis.

## step 6 - INTERPRETATION

The interpretation phase, although closely related to the evaluation phase, includes a more fine-grained understanding of the latent variables. The main goal of the interpretation 
phase is to go beyond the latent variables and understand the latent variables in the context of the domain under study. This phase is highly depending on the research question 
that we would want to have answered. What topics are present, how are they distributed over time, and how are they related to other topics are possible ways to explore the output 
of the LDA analysis. Similarly to the evaluation phase, aiming for a deeper understanding of the topics might also result in flaws in the analysis. For example, a visualization of 
the topics that places two very distinct topics in close proximity, high probability of a topic in a document that does not cover aspects of that topic, or topics that should not 
co-occur together are indicators of flaws or areas of improvements. In such cases, it would be wise to revisit the pre-processing phase and to re-run the analysis with, for instance, 
different model parameters or pre-processing steps.

# Reference Publications

*	Syed, S., Borit, M., & Spruit, M. (2018). Narrow lenses for capturing the complexity of fisheries: A topic analysis of fisheries science from 1990 to 2016. Fish and Fisheries, 19(4), 643–661. http://doi.org/10.1111/faf.12280
*	Syed, S., & Spruit, M. (2017). Full-Text or Abstract? Examining Topic Coherence Scores Using Latent Dirichlet Allocation. In 2017 IEEE International Conference on Data Science and Advanced Analytics (DSAA) (pp. 165–174). Tokyo, Japan: IEEE. http://doi.org/10.1109/DSAA.2017.61
*	Syed, S., & Spruit, M. (2018a). Exploring Symmetrical and Asymmetrical Dirichlet Priors for Latent Dirichlet Allocation. International Journal of Semantic Computing, 12(3), 399–423. http://doi.org/10.1142/S1793351X18400184
*	Syed, S., & Spruit, M. (2018b). Selecting Priors for Latent Dirichlet Allocation. In 2018 IEEE 12th International Conference on Semantic Computing (ICSC) (pp. 194–202). Laguna Hills, CA, USA: IEEE. http://doi.org/10.1109/ICSC.2018.00035
*	Syed, S., & Weber, C. T. (2018). Using Machine Learning to Uncover Latent Research Topics in Fishery Models. Reviews in Fisheries Science & Aquaculture, 26(3), 319–336. http://doi.org/10.1080/23308249.2017.1416331




