#!/usr/bin/env python
# coding: utf-8

# # Dimensionality reduction and visualization with polarity based color-coding Using TSNE

# In this notebook we will do dimensionality reduction and visualization using TSNE over 'Text_type data' named Amazon Fine Food reviews . In the process we will explore some of the interesting text featurization techniques of NLP like bag of words, TF-IDF, Avg Word2Vec, TF-IDF Word2vec, Glove etc.

# **Overview of Data**
# 
# This dataset consists of reviews of fine foods (food products) from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plaintext review. We also have reviews from all other Amazon categories.<br>
# 
# Data Source : https://snap.stanford.edu/data/web-FineFoods.html or https://www.kaggle.com/snap/amazon-fine-food-reviews (not using the updated one, using the old one here from the Stanford website)
# 
# Paper reference : http://i.stanford.edu/~julian/pdfs/recsys13.pdf
# 
# Number of reviews: 568,454<br>
# Number of users: 256,059<br>
# Number of products: 74,258<br>
# Timespan: Oct 1999 - Oct 2012 (13 years)<br> 
# Number of Attributes/Columns in data: 10<br>
# 
# Attribute Information:<br>
# 
# >1.Id<br>
# 
# >2.ProductId - unique identifier for the product<br>
# 
# >3.UserId - unqiue identifier for the user<br>
# 
# >4.ProfileName<br>
# 
# >5.HelpfulnessNumerator - number of users who found the review     helpful<br>
# 
# >6.HelpfulnessDenominator - number of users who indicated whether they<br> found the review helpful or not<br>
# 
# >7.Score - rating between 1 and 5<br>
# 
# >8.Time - timestamp for the review<br>
# 
# >9.Summary - brief summary of the review<br>
# 
# >10.Text - text of the review<br>

# **Machine Learning Problem Statement**<br>
# 
# This is a Typical Sentiment Analysis Binary classification machine learning problem. We still don't know if the data is balanced or imbalanced over class.We have to do EDA to find that out. 
# 
# Given a review, determine whether the review is positive (Rating of 4 or 5) or negative (rating of 1 or 2).
# 
# [Q] How to determine if a review is positive or negative?
# 
# [Ans] We could use the Score/Rating. A rating of 4 or 5 could be cosnidered a positive review. A review of 1 or 2 could be considered negative. A review of 3 is nuetral and ignored. This is an approximate and proxy way of determining the polarity (positivity/negativity) of a review.

# **Loading the Data**
# 
# The dataset is available in two forms
# 
# >    1) .csv file<br>
# >    2) SQLite Database<br>
# 
# Here as we only want to get the global sentiment of the recommendations (positive or negative), we will purposefully ignore all Scores equal to 3. If the score id above 3, then the recommendation wil be set to "positive". Otherwise, it will be set to "negative".

# In[1]:


# Some required library imports 

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")



import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


# In[2]:


get_ipython().system('python -m pip install -U gensim')


# In[3]:


import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

from tqdm import tqdm
import os


# **Reading Data**

# In[4]:


reviews = pd.read_csv("C:\\Users\\User\\Desktop\\reviews.csv" )
reviews.head()


# In[5]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 9999;')


# In[6]:


# Give reviews with Score > 3 a positive rating, and reviews with a score < 3 a negative rating.

def partition(x):
    if x<3:
        return 0
    return 1

actualscore = reviews['Score']
newscores = actualscore.map(partition)
reviews['Score'] = newscores
print('number of data points in data',reviews.shape)
reviews.head(25)


# **Grouping Data by Score**

# In[7]:


v = reviews.groupby('Score').size().reset_index(name = 'count')
grpbyscore = pd.DataFrame(v)
grpbyscore


# **Data Cleaning : Deduplication**
# 
# Data Cleaning is an extremely important aspect in real world Machine Learning. Any Machine Learning system you design is not a God by itself. It simply obeys the universal rule of **"Garbage in - Garbage out**"

# The data is imbalanced.

# **Grouping Data by unique User ID**

# In[8]:


mp = pd.DataFrame(reviews)
mp


# **How many unique users are present?**

# In[9]:


k = mp.groupby('UserId').size().reset_index(name = 'count')
grpbyuserid = pd.DataFrame(k)
grpbyuserid


# How many users are exist more than once ? 

# In[10]:


v = grpbyuserid[grpbyuserid['count'] >1]
v


# In[11]:


v['count'].sum()


# What's the maximum number of times a user has been repeated ?

# In[12]:


v['count'].max()


# Are the productIds also repeated for all the 448 repeated UserId?

# In[13]:


v[v['count'] == 448]


# In[14]:


ac = mp[mp['UserId'] == 'A3OXHLG6DIBRW8']
ac


# The productIds have not been repeated for the same userId. Hence, there is a considerable chance all repeated Userids need not have repeated ProductIds as well. <br>
# 
# May be , the same user is buying different products on amazon. This possibility has to be considered by us.<br>
# 
# UserIds are not being repeated for the same product Ids<br>

# But, There may exist repeated some user Ids for which the product Ids are also being repeated. They are duplicates of data and lets detect below if they are existing or not ----------

# It has been observed from above that users are being repeated. But the real question is are the usersIds repeated for the same products ? <br>
# 
# Is there real duplication in rows in the above data frame ?

# In[15]:


o = mp.groupby(['UserId','ProductId']).size().reset_index(name = 'count')
grpbyuseridandproductid = pd.DataFrame(o)
grpbyuseridandproductid


# In[16]:


lm = grpbyuseridandproductid[grpbyuseridandproductid['count'] > 1]
lm


# In[17]:


grpbyuseridandproductid[grpbyuseridandproductid['count'] > 10]


# **At max how many repitions have occured?**

# In[18]:


grpbyuseridandproductid['count'].max()


# From the above we can notice that there are duplicates rows existing in our data. But are they really duplicates? Does repition of user ids along with the repition of corrsponding Product ids imply presence of duplicate rows? <br> answer is **Yes**
# 
# But the real question we have to ask is , are non-repeting productIds for the same user, refering to the same product on amazon.com or different products? <br>
# 
# Is presence of  different product Ids enough to conclude that the products are different ?

# It is observed (as shown in the table below) that the reviews data had many duplicate entries. Hence it was necessary to remove duplicates in order to get unbiased results for the analysis of the data.  Following is an example:

# In[19]:


ac = mp[mp['UserId'] == 'A3OXHLG6DIBRW8']
ac


# In[20]:


lk = mp[mp['ProductId'] == 'B005K4Q1VI']
lk


# In[21]:


lk[lk['ProfileName'] == 'C. F. Hill "CFH"']


# Like observed above for product Id == 'B005K4Q1V'  and ProfileName == (C. F. Hill "CFH"	), there exists 2 repeated entries, which are mere duplicates of each other . <br>
# 
# We have to remove such entries. Similarly ----------------

# In[22]:


kv = mp[mp['ProfileName'] == 'Geetha Krishnan']
kv


# We will remove duplicates of entries like above before doing meaningful EDA on our data<br>As can be seen above the same user has multiple reviews with the same values for HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary and Text and on doing analysis it was found that : 
# 
# ProductId=B000HDOPZG was Loacker Quadratini Vanilla Wafer Cookies, 8.82-Ounce Packages (Pack of 8)
# 
# ProductId=B000HDL1RQ was Loacker Quadratini Lemon Wafer Cookies, 8.82-Ounce Packages (Pack of 8) and so on
# 
# It was inferred after analysis that reviews with same parameters other than ProductId belonged to the same product just having different flavour or quantity. Hence in order to reduce redundancy it was decided to eliminate the rows having same parameters.
# 
# **The method we will be using to remove duplicates is that we first sort the data according to ProductId and then just keep the first similar product review and delete the others. for eg. in the above just the review for ProductId=B000HDL1RQ remains. This method ensures that there is only one representative for each product and deduplication without sorting would lead to possibility of different representatives still existing for the same product.**

# In[23]:


#Sorting data according to ProductId in ascending order
sorted_data = mp.sort_values('ProductId',axis = 0, ascending = True, inplace = False, kind = 'quicksort', na_position = 'last')
sorted_data


# In[24]:


final_data = sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Summary","HelpfulnessNumerator","HelpfulnessDenominator","Text"}, keep='first', inplace=False)
final_data


# In[25]:


#Checking to see how much % of data still remains

(final_data['Id'].size*1.0)/(reviews['Id'].size*1.0)*100


# In[26]:


#Checking to see how much % of data still remains

(final_data['Id'].size*1.0)/(mp['Id'].size*1.0)*100


# **Detecting Rows where HelpfulnessNumerator is greater than HelpfulnessDenominator**

# In[27]:


final_data[final_data['HelpfulnessNumerator'] > final_data['HelpfulnessDenominator']]


# We have to remove entries like above because HelpfulnessNumerator is greater than HelpfulnessDenominator which is not practically possible .<br>
# 
# We will remove the above 2 rows from the data frame.

# In[28]:


final_data = final_data[final_data.HelpfulnessNumerator <= final_data.HelpfulnessDenominator]


# In[29]:


final_data


# **Comparing original data and preprocessed data**

# In[30]:


print(reviews.shape)
print(final_data.shape)


# **Class label distribution over preprocessed data**

# In[31]:


final_data.groupby('Score').size().reset_index(name = 'count')


# Data is still imbalanced

# **Text Preprocessing on Review Text**

# Now that we have finished deduplication our data requires some text preprocessing before we go on further with analysis.
# 
# Hence in the Preprocessing phase we do the following in the order below:-
# >1. Begin by removing the html tags
# >2. Remove any punctuations or limited set of special characters like ,        or . or # etc.
# >3. Check if the word is made up of english letters and is not alpha-          numeric
# >4. Check to see if the length of the word is greater than 2 (as it was        researched that there is no adjective in 2-letters)
# >5. Convert the word to lowercase
# >6. Remove Stopwords
# >7. Finally Snowball Stemming the word (it was obsereved to be better than Porter Stemming)
# >8. After which we collect the words used to describe positive and negative reviews
# 
# After which we collect the words used to describe positive and negative reviews

# **Class distributions over the sampled data**

# In[32]:


V =final_data.groupby('Score').size().reset_index(name = 'count')
V


# **Printing some random reviews from smapled data**

# In[33]:


r_0 = final_data['Text'].values[0]
print(r_0)

print("="*60)

r_1000 = final_data['Text'].values[1000]
print(r_1000)
print("="*60)

r_10k = final_data['Text'].values[10000]
print(r_10k)
print("="*60)

r_31567 = final_data['Text'].values[31567]
print(r_31567)
print("="*60)


# Before proceeding further to text pre processing and applyling text featurization techniques of NLP, let us first do some basic EDA on our final_sample here

# **Distribution of Ratings**

# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.available


# **Distribution plot over class label in final_data**

# In[35]:


get_ipython().run_cell_magic('time', '', "import seaborn as sn\nsn.distplot(final_data['Score'],kde = False,bins = 10, color = 'green')\nplt.style.use('seaborn-darkgrid')\nplt.show()\nplt.rcParams['figure.figsize'] = (10,10)")


# **KDE plot over class label in the final_data**

# In[36]:


get_ipython().run_cell_magic('time', '', "sn.kdeplot(final_data['Score'],cumulative  = True , color = 'red',shade = True, bw = .1,label = 'bw = 0.1')\nsn.kdeplot(final_data['Score'],cumulative  = True , color = 'indigo',shade = True, bw = .4,label = 'bw = 0.4')\nsn.kdeplot(final_data['Score'],cumulative  = True , color = 'green',shade = True, bw = .6,label = 'bw = 0.6')\nsn.kdeplot(final_data['Score'],cumulative  = True , color = 'orange',shade = True, bw = 1, label = 'bw = 1')\nplt.rcParams['figure.figsize'] = (10,10)")


# **Script for Removing HTML tags**

# In[37]:


#https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string

import re

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
  return cleanr.sub(' ',raw_html)
  
testing = cleanhtml('<a href="foo.com" class="bar">I Want This <b>text!</b></a><>')
print(testing)


# **Script for Removing Punctuations**

# In[38]:


# https://www.geeksforgeeks.org/removing-punctuations-given-string/
import string
string.punctuation

def remove_punc(txt):
    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct


# **Script for Removing URLs**

# In[39]:


# https://gist.github.com/MrEliptik/b3f16179aa2f530781ef8ca9a16499af
import re 
def remove_URL(sample):
    """Remove URLs from a sample string"""
    p = re.compile('http\S+')
    return p.sub('', sample)
check = remove_URL('This data set is taken from  https://snap.stanford.edu/data/web-FineFoods.html or https://www.kaggle.com/snap/amazon-fine-food-reviews ')
print(check)

print(remove_URL('The research paper for this project is  http://i.stanford.edu/~julian/pdfs/recsys13.pdf'))


# **Script for Removing Words with Numbers**

# In[40]:


def remove_words_withnumbers(sample):
    
    """Remove URLs from a sample string"""
    q = re.compile("\d")
    return q.sub(' ', sample)
working = remove_words_withnumbers('deeplearning ksjdaskj 34i290480')
print(working)


# **Script for Removing Special Characters from the words**

# In[41]:


def remove_specialcharacters(sample):
    """Remove Special characters from a sample string"""
    gh = re.compile("[^A-Za-z0-9]+")
    return gh.sub(' ',sample)
working = remove_specialcharacters('abd 1%2&**^$336363#886#')
print(working)


# **Script to Convert Upper case to Lower case**

# In[42]:


def to_Lowercase(string):
    return string.lower()
print(to_Lowercase('SrK'))
print(to_Lowercase('kvPY'))


# **Removing Multiple Spaces**

# In[43]:


def remove_multiplespace(text):
    jk = re.compile(r"\s+")
    return jk.sub(' ',text)
klm = remove_multiplespace('The film      Pulp Fiction      was released in   year 1994.')
print(klm)


# **Removing Stop Words**

# In[44]:


pip install --user -U nltk


# In[45]:


import nltk
nltk.download('stopwords')


# In[46]:


from nltk.corpus import stopwords
stop = stopwords.words('english')
print(stop)

# make a list of all common negative words and make sure they are not removed, since they
# are an important aspect to determine neagtive sentiment of the reviews
excluding = ['against','not','don', "don't",'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',"didn't",
'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
"isn't",
'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",'shouldn', "shouldn't", 'wasn',
"wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

new_stops = [words for words in stop if words not in excluding]
print(new_stops)


# In[47]:


texts_original = pd.DataFrame(final_data['Text'])
texts_original


# In[48]:


print(f'Input data has {len(texts_original)} rows, {len(texts_original.columns)} columns')


# In[49]:


pd.set_option('display.max_colwidth',4000)
texts_original


# In[50]:


texts_original = texts_original.rename(columns = {'Text':'Original_Review_Text'})
texts_original


# In[51]:


get_ipython().run_cell_magic('time', '', "texts_original['Htmltags_removed'] = texts_original['Original_Review_Text'].apply(lambda x: cleanhtml(x))\ntexts_original['URLs_removed'] = texts_original['Htmltags_removed'].apply(lambda x: remove_URL(x))\ntexts_original['Removed_Words_with_numbers'] = texts_original['URLs_removed'].apply(lambda x: remove_words_withnumbers(x))\ntexts_original['Removed_Words_with_Special_Characters'] = texts_original['Removed_Words_with_numbers'].apply(lambda x: remove_specialcharacters(x))\ntexts_original['Removed_Punctuations'] = texts_original['Removed_Words_with_Special_Characters'].apply(lambda x: remove_punc(x))\ntexts_original['Converted_to_Lowercase'] = texts_original['Removed_Punctuations'].apply(lambda x: to_Lowercase(x))\ntexts_original")


# **Tokenization**

# In[52]:


get_ipython().run_cell_magic('time', '', "import re\n\ndef tokenize(txt):\n    tokens = re.split('\\W+',txt)\n    return tokens\n\ntexts_original['Tokenization_applied'] = texts_original['Converted_to_Lowercase'].apply(lambda x: tokenize(x.lower()))\n\ntexts_original")


# **Remove Stopwords**

# In[53]:


get_ipython().run_cell_magic('time', '', "def remove_stopwords(tokenized_texts):\n    txt_clean = [word for word in tokenized_texts if word not in new_stops]\n    return txt_clean\n\ntexts_original['Stopwords_removed'] = texts_original['Tokenization_applied'].apply(lambda x: remove_stopwords(x))\n  \ntexts_original")


# **Porter Stemming**

# In[54]:


import nltk
from nltk.stem import PorterStemmer 
ps = PorterStemmer()
dir(ps)


# In[55]:


def porter_stem (txt):
    text = [ps.stem(word) for word in txt]
    return text


# In[56]:


get_ipython().run_cell_magic('time', '', "texts_original['Porter_stemmed'] = texts_original['Stopwords_removed'].apply(lambda x : porter_stem(x))\n\ntexts_original")


# **Snowball Stemming**

# In[57]:


from nltk.stem import SnowballStemmer
ss = SnowballStemmer('english')
dir(ss)


# In[58]:


def snowball_stem(txt):
    result = [ss.stem(word) for word in txt]
    return result


# In[59]:


get_ipython().run_cell_magic('time', '', "\ntexts_original['Snowball_stemmed'] = texts_original['Stopwords_removed'].apply(lambda x : snowball_stem(x))\n\ntexts_original")


# In[60]:


#with open('mytable.tex','w') as tf:
#    tf.write(texts_original.head().to_latex())


# **Word Net Lemmatization**

# In[61]:


import nltk 
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()
dir(wn)


# In[62]:


get_ipython().run_cell_magic('time', '', "print(ss.stem('goose'))\nprint(ss.stem('geese'))")


# In[63]:


get_ipython().run_cell_magic('time', '', "print(wn.lemmatize('goose'))\nprint(wn.lemmatize('geese'))")


# From the above we can see that Lemmatization is slower than stemming , but it's much more accurate than steeming

# **Lemmatization**

# In[64]:


def lemmatize(text):
    fgh = [wn.lemmatize(word) for word in text]
    return fgh


# In[65]:


get_ipython().run_cell_magic('time', '', "\ntexts_original['Wordnet_Lemmatized'] = texts_original['Stopwords_removed'].apply(lambda x : lemmatize(x))\n\ntexts_original")


# **Unigram only Bag Of Words**

# In[66]:


final_data.shape


# In[67]:


texts_original.shape


# In[68]:


new_df = texts_original[['Converted_to_Lowercase','Tokenization_applied','Stopwords_removed','Wordnet_Lemmatized']]
new_df


# In[69]:


def join(txt):
    txt_joined = " ".join([c for c in txt])
    return txt_joined


# In[70]:


get_ipython().run_cell_magic('time', '', "new_df['Final_Preprocessed_words'] = new_df['Wordnet_Lemmatized'].apply(lambda x : join(x))\nnew_df")


# **Count Vectorization - Unigram BOW on Limited no. of Reviews**

# In[71]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

corpus = new_df['Final_Preprocessed_words'].values[0:10]

x = cv.fit(corpus)
print(x.vocabulary_)
print(cv.get_feature_names())

y = cv.transform(corpus)
print(y.shape)
print(y.toarray())


# In[72]:


print(y)

subdf = pd.DataFrame(y.toarray(), columns = cv.get_feature_names())
subdf


# **Count Vectorizer on All Review Texts - Unigrams**

# In[73]:


get_ipython().run_cell_magic('time', '', "from sklearn.feature_extraction.text import CountVectorizer\ncv1 = CountVectorizer()\n\ncorpus1 = new_df['Final_Preprocessed_words']\nx = cv1.fit(corpus1)")


# In[131]:


dir(cv1)


# In[74]:


# print(x.vocabulary_) - for getting the frequency count of each word of Uni gram Model


# In[75]:


# print(cv1.get_feature_names()) - for getting name of each unique feature in Unigrams Model


# In[76]:


fc = cv1.fit_transform(corpus1)


# In[77]:


print(fc.shape)


# There are 98k unique set of words like it can be observed above 

# **Bi-gram and N-gram Count Vectorization - BOW**

# In[78]:


from sklearn.feature_extraction.text import CountVectorizer
cv2 = CountVectorizer(ngram_range = (2,2))

dox = new_df['Final_Preprocessed_words'].values[0:10]

    
z = cv2.fit(corpus)
print(z.vocabulary_)
print(cv2.get_feature_names())

xy = cv2.transform(corpus)
print(xy.shape)
print(xy.toarray())


# In[79]:


print(xy)

subdf1 = pd.DataFrame(xy.toarray(), columns = cv2.get_feature_names())
subdf1


# **Count Vectorizer on All Review Texts - Bigrams Only**

# In[80]:


get_ipython().run_cell_magic('time', '', "from sklearn.feature_extraction.text import CountVectorizer\ncv3 = CountVectorizer(ngram_range = (2,2))\n\ncorpus2 = new_df['Final_Preprocessed_words']\nty = cv3.fit_transform(corpus2)")


# In[81]:


print(ty.shape)


# In[82]:


print(cv3.get_feature_names())


# **Generating Word Clouds on entire Data**

# In[83]:


import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# In[84]:


get_ipython().run_cell_magic('time', '', '\ntext = " ".join(txt for txt in new_df.Final_Preprocessed_words)\nprint ("There are {} words in the combination of all review.".format(len(text)))\n\n# Create and generate a word cloud image:\nwc = WordCloud(background_color="white",width=1600, height=800).generate(text)')


# In[85]:


# https://www.datacamp.com/community/tutorials/wordcloud-python

plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.rcParams['figure.figsize'] = (50,40)
plt.tight_layout(pad=0)


# **Tri - Grams**

# In[86]:


from sklearn.feature_extraction.text import CountVectorizer
cv_3 = CountVectorizer(ngram_range = (3,3))

files = new_df['Final_Preprocessed_words'].values[0:10]


ui = cv_3.fit(corpus)
print(ui.vocabulary_)
print(cv_3.get_feature_names())

xyz = cv_3.transform(corpus)
print(xyz.shape)
print(xyz.toarray())


# In[87]:


print(xyz)

subdf2 = pd.DataFrame(xyz.toarray(), columns = cv_3.get_feature_names())
subdf2


# In[88]:


get_ipython().run_cell_magic('time', '', "from sklearn.feature_extraction.text import CountVectorizer\ncv4 = CountVectorizer(ngram_range = (3,3))\n\ncorpus_3 = new_df['Final_Preprocessed_words']\ntyq = cv_3.fit_transform(corpus_3)")


# In[89]:


print(tyq.shape)


# In[90]:


our_df = new_df[['Wordnet_Lemmatized','Final_Preprocessed_words']]
our_df


# In[91]:


wer = pd.DataFrame(final_data['Score'])
wer


# In[92]:


our_df['Score'] = wer['Score']


# In[93]:


our_df


# In[94]:


neg = our_df[our_df['Score'] == 0]
neg


# **Word Cloud on Negative words**

# In[106]:


get_ipython().run_cell_magic('time', '', '\nfly = " ".join(txt for txt in neg.Final_Preprocessed_words)\nprint ("There are {} words in negative reviews.".format(len(fly)))\n\n# Create and generate a word cloud image:\ncloud = WordCloud(background_color="white",width=1600, height=800).generate(fly)')


# In[96]:


plt.imshow(cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.rcParams['figure.figsize'] = (50,40)
plt.tight_layout(pad=0)


# In[97]:


pos = our_df[our_df['Score'] == 1]
pos


# **Word Cloud on Positive Words**

# In[107]:


get_ipython().run_cell_magic('time', '', '\nyolo = " ".join(txt for txt in pos.Final_Preprocessed_words)\nprint ("There are {} words in positive reviews.".format(len(yolo)))\n\n# Create and generate a word cloud image:\ncloud_1 = WordCloud(background_color="white",width=1600, height=800).generate(yolo)')


# In[99]:


plt.imshow(cloud_1, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.rcParams['figure.figsize'] = (50,40)
plt.tight_layout(pad=0)


# In[112]:


from collections import Counter
print('No of Positive words :',len(yolo))
print('No of negative words :',len(fly))

positive = Counter(yolo)
negative = Counter(fly)

print('\nMost Common words in positive reviews are :',positive.most_common(15))
print('\nMost Common words in negative reviews are :',negative.most_common(15))


# from matplotlib.pyplot import figure
# figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
# pos_words = positive.most_common(15)
# pos_words.sort(key=lambda x: x[1], reverse=False)
# words=[]
# times=[]
# for w,t in yolo:
# words.append(w)
# times.append(t)
# plt.barh(range(len(words)),times)
# plt.yticks(range(len(words)),words)
# plt.xlabel('Most Popular Positive Words')
# plt.show()

# **TF - IDF**

# In[135]:


from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(ngram_range = (1,2))

files = our_df['Final_Preprocessed_words'].values[0:10]

lionel = tv.fit(files)
print(lionel.vocabulary_)
print(lionel.get_feature_names())

pqr = tv.transform(corpus)
print(pqr.shape)
print(pqr.toarray())


# In[136]:


print(pqr)
tf_frame = pd.DataFrame(pqr.toarray(),columns = tv.get_feature_names())
tf_frame


# **Tf - Idf on all Reviews**

# In[155]:


get_ipython().run_cell_magic('time', '', "from sklearn.feature_extraction.text import TfidfVectorizer\ntv2 = TfidfVectorizer(ngram_range = (1,2))\n\nthings = our_df['Final_Preprocessed_words'].values\niol = tv2.fit_transform(things)  ")


# In[164]:


print(iol.shape)


# In[165]:


dir(tv2)


# In[166]:


sdf = tv2.get_feature_names()
len(sdf)


# In[167]:


sdf[1000:1009]


# In[170]:


def top_tfidf_features(row, feature, top_n = 25):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    data_frame = pd.DataFrame(top_feats)
    data_frame.columns = ['Feature','Tf - Idf']
    return data_frame

top_tfidf = top_tfidf_features(things[1,:].toarray()[0], features,25)
    
    


# In[171]:


our_df


# In[174]:


token_count = sum(len(word) for word in our_df['Wordnet_Lemmatized'])
print('There are total of {} tokens our reviews text'.format(token_count))


# **Train W 2 Vec**

# In[187]:


from gensim.models import Word2Vec
from gensim.models import KeyedVectors

w2v = Word2Vec()
dir(w2v)


# In[191]:


kv = KeyedVectors
dir(kv)

