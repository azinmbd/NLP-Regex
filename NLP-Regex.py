#!/usr/bin/env python
# coding: utf-8

# In[15]:


import nltk 


# In[16]:


import re


# In[17]:


nltk.download('words')


# In[18]:


wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]


# In[19]:


[w for w in wordlist if re.search('ed$', w)]
# $ means end ed$ meand edns with ED// [] means that we are seeking a list of words in output


# In[20]:


[w for w in wordlist if re.search('^..j..t..$', w)]
#has 8 letter that has j and t ---- ^ means nothing comes before it


# In[21]:


[w for w in wordlist if re.search('-?mail$', w)]
#everything before ? in optional Matches zero or one occurrence


# In[22]:


[w for w in wordlist if re.search('^..mail$', w)]
# ^ + .. exact lenght of letters


# In[23]:


sum(1 for w in wordlist if re.search('-?mail$', w))


# In[24]:


[w for w in wordlist if re.search('^[ghi][mno][jlk][def]$', w)]
# 4 letters becase we have 4 pairs of [] # ^ means nothing before


# In[25]:


[w for w in wordlist if re.search('[ghi][mno][jlk][def]$', w)]


# In[26]:


nltk.download('nps_chat')


# In[27]:


chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
# for itterates over the words in nps -- set gets the unique words -- sorted sorts the data


# In[28]:


[w for w in chat_words if re.search('^m+i+n+e+$', w)]
# +: Matches one or more occurrences of the preceding character


# In[29]:


[w for w in chat_words if re.search('^[ha]+$', w)]
# starts with h or a and anything before can happen as many time


# In[30]:


[w for w in chat_words if re.search('^m*i*N*e*$', w)]
# *: Matches zero or more occurrences for the letter that comes before 


# In[31]:


[w for w in chat_words if re.search('^m*i*N*e$', w)]
# *: Matches zero or more occurrences 


# In[32]:


nltk.download('treebank')


# In[33]:


wsj = sorted(set(nltk.corpus.treebank.words()))


# In[34]:


[w for w in wsj if re.search('^[0-9]+\.[0-9]+$', w)]
# \ means everything after is mandatory like the .


# In[35]:


[w for w in wsj if re.search('^[A-Z]+\$$', w)]
# \$ mean that $ must happen in text


# In[36]:


[w for w in wsj if re.search('^[0-9]{4}$', w)]
# the lenght shoule be 4


# In[37]:


[w for w in wsj if re.search('^[0-9]+\-[a-z]{3,5}$', w)]
# - is mandatory because of \ 3,5 at leat 3 and max 5 in lenght


# In[38]:


[w for w in wsj if re.search('^[0-9]+-[a-z]{3,5}$', w)]
# same as above we should use \ for the charcter's like +, * , $ that have meaning


# In[39]:


[w for w in wsj if re.search('^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}$', w)]
#{5,} at leat 5 characters {,6} maxmimom 6 chacters


# In[40]:


[w for w in wsj if re.search('(ed|ing)$', w)]
# | means or we have ed or ing at the end


# In[41]:


word = "supercalifragilisticexptialidocious"


# In[42]:


re.findall('[aeiou]', word)


# In[43]:


len(re.findall('[aeiou]', word))


# In[44]:


fd = nltk.FreqDist(vs for word in wsj
                  for vs in re.findall('[aeiou]{2,}', word))
# showing the pair and the number of occurance


# In[45]:


fd.items()


# In[46]:


nltk.download('toolbox')


# In[47]:


rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')


# cvs = [cv for w in rotokas_words for cv in re.findall('[ptksvr][aeiou]', w)]

# In[48]:


cvs = [cv for w in rotokas_words for cv in re.findall('[ptksvr][aeiou]', w)]


# In[49]:


cdf = nltk.ConditionalFreqDist(cvs)


# In[50]:


cdf.tabulate()


# In[51]:


cv_word_pairs = [(cv , w) for w in rotokas_words for cv in re.findall('[ptksvr][aeiou]', w)]


# In[52]:


cv_index = nltk.Index(cv_word_pairs)


# In[53]:


cv_index['su']


# In[54]:


cv_index['po']


# In[55]:


def stem(word):
    for suffix in ['ing', 'ly', 'ed', 'ious', 'ive', 'es', 's', 'ment']:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word


# In[56]:


list1 = ['laptops', 'processing', 'loudly']


# In[57]:


for w in list1:
    print(stem(w))


# In[58]:


re.findall('^.*(ing|ly|ed|ious |ies|ive|es|s❘ment)$', 'processing')
# .*: Matches any character (except newline) zero or more times.


# In[59]:


re.findall('^.*(?:ing|ly|ed|ious |ies|ive|es|s❘ment)$', 'processing')
# ?: anything before is not mandatory


# In[60]:


re.findall('^(.*)(ing|ly|ed|ious |ies|ive|es|s❘ment)$', 'processing')
# (.*)show what happens before that


# In[61]:


re.findall('^(.*)(ing|ly|ed|ious |ies|ive|es|s❘ment)$', 'processes')


# In[62]:


re.findall('^(.*?)(ing|ly|ed|ious |ies|ive|es|s❘ment)?$', 'language')


# In[63]:


text = "I liked Python's processing in the intersting NLP classes."
words = nltk.tokenize.word_tokenize (text)


# In[64]:


porter = nltk.PorterStemmer()


# In[65]:


lancaster = nltk.LancasterStemmer()


# In[66]:


[porter.stem(t) for t in words]


# In[67]:


[lancaster.stem(t) for t in words]


# In[68]:


wnl = nltk.WordNetLemmatizer()


# In[69]:


nltk.download('wordnet')


# In[70]:


[wnl.lemmatize(t) for t in words]


# In[71]:


nltk.download('averaged_perceptron_tagger')


# In[96]:


text = "I love open source"
words = nltk.tokenize.word_tokenize(text)
words_tagged = nltk.pos_tag(words)


# In[97]:


grammer = "NP: {<JJ><NN>}"


# In[98]:


from nltk.chunk import RegexpParser


# In[99]:


parser = RegexpParser(grammer)
tree = parser.parse(words_tagged)
tree.draw()


# In[100]:


text = "I like Python and interesting NLP classs."
words = nltk.tokenize.word_tokenize(text)
words_tagged = nltk.pos_tag(words)

grammer = "NP: {<JJ><NN>}"

parser = RegexpParser(grammer)
tree = parser.parse(words_tagged)
tree.draw()


# In[ ]:




