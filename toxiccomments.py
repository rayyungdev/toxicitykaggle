#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import sklearn


# # What we know:
# 
# #### The kaggle provided dataset contains 3 files:  
# ##### comment_to_score.csv
#     - Contains all unranked comments. Our objective is to rank these comments based off the toxicity, with 1 being the most toxic. 
# 
# ##### sample_submission.csv
#     - Contains an example of how the submission should look like... Not sure if this will help with training?
# 
# ##### validation_data.csv
#     - According to kaggle this is the "pair rankings that can be used to validate models. Includes annotator work id and how the annotator ranked a given pair of comments"
#     
# # Understanding the objective
# Before moving on, let's take a moment and understand what our overall objective is: 
# We want to rank a list of comments base off of the level of toxicity without the use of any provided training data. However, while we do not have training data, we have provided validation data, which includes a set of toxicity rankings that might help to validate models.
# 
# # Understanding our data:
# In the next few lines, we will import our data and see what we're actually dealing with and see if the data provided is actually useful. 

# #### comments_to_score.csv
#         - We knew how the data looked like before, but let's take a closer look and see what might need to be cleaned up  
#    
#    - Column Names
#        - comment_id (comment identifier)
#        - text (actual text)

# In[2]:


# comments_to_score data
df = pd.read_csv("./data/comments_to_score.csv")
eg1 = df.text[0]
print(eg1)


# In[3]:


eg2 = df.text[25]
print(eg2)


# ##### took two comments from the provided dataset. We can make a few observations based off of the content of the texts:   
#   
# - First, punctuation. Normally, when we deal with datasets that involve texts, we would ignore punctuation. Howevever, knowing the habits of the internet, the punctuation can reflect toxicity. For instance, quotation marks have been used to represent sarcasm.
#     -Building onto this, this same concept applies with capitalized letters. Generally speaking, when dealing with texts, we would try to make the letters consistent. However, given the habits of the internet, we can 
#     
# - I believe we can consider the second comment to be more toxic. But let's try to understand what makes it toxic: 
#     - "If you weren't so lazy and intent on harassment, you could use Google to search for ""Gargamel"" and ""Jew""". 
#         - This particular sentence feels targeted, with lazy being the negative description.   
#         - Might want to consider pulling common words, phrases, and general patterns with the provided validation dataset as a means to start. However, keep in mind that the dataset also contains data that is not in our testing set

# ###### Validation_data.csv
# After playing around with the data, I've come to realize the order of the comments do not matter. As described on Kaggle, this ia a collection of comments, which a person judged a pair of comments to see which one is more or less toxic (making almost binary). The comments are kinda hilarious from a third person perspective.  
#   
# Looking at the data on excel, I noticed that some comments are repeated, showing that it's been consistently rated as being more toxic than the other. It might be interesting to make a histogram based off of the number times a comment is rated as toxic. We can also create a list of toxic comments and have it ordered from most toxic to least toxic based off the number of counts it was rated as more toxic. 

# In[4]:


vd = pd.read_csv("./data/validation_data.csv")
eg3 = vd.loc[0,:]
eg3


# In[5]:


eg4 = vd.loc[100,:]
eg4


# ##### sample_submission.csv
# This simply tells us how we should make our submission, it is interesting that we only care about the comment_id and the score. Although I am more interested in how the score will be developed. Is it unique or what?

# In[6]:


eg5 = pd.read_csv("./data/sample_submission.csv")
eg5.loc[0:5,:]


# ### Testing out Ideas
# 
# In the previous section, I discussed ideas to explore. In this section, we will try out those ideas and see how they could apply to our model. 

# In[7]:


## Find all the unique comments in the validation dataset. 

# Alright, so I want to look at this data with ease. To save time and memory, I'm going to give each unique comment a unique number
# And count the number of occurences the unique ID has. 
toxic_unique = vd.more_toxic.unique()
toxic_dict = {};
not_toxic_unique = vd.less_toxic.unique()
not_toxic_dict = {};
count = 0;

for tword in toxic_unique:
    toxic_dict[count] = [tword, len(vd.index[vd.more_toxic == tword].tolist())]
    count += 1
    # Create a dictionary that we can use
    #I don't know how much time or memory this uses, but basicaly I am creating a dictionary which stores the number of counts per unique comment

count = 0
for ntword in not_toxic_unique:
    not_toxic_dict[count] = [ntword, len(vd.index[vd.less_toxic == ntword].tolist())] # Create a dictionary that we can use
    count +=1


# In[8]:


toxic_df = pd.DataFrame.from_dict(toxic_dict, orient = 'index', columns = ['text', 'count']).sort_values(by =['count'], ascending = False)
not_toxic_df = pd.DataFrame.from_dict(not_toxic_dict, orient = 'index', columns = ['text', 'count']).sort_values(by = ['count'], ascending = False)

toxic_df[0:20]


# In[9]:


not_toxic_df[0:20]


# In[10]:


print("toxic_df length: ", len(toxic_df))
print("not_toxic_df length: ", len(not_toxic_df))


# As you can see, I created two different dataframes to rank the frequency of comments found in the toxic and less toxic columns from the validity data provided. One thing to keep in mind is that we will want our algorithm to generalize, therefore having a keyword classifer might not be particularly useful. Instead of looking for specific words, we should think about the underlying trends in the data.      
#   
# Interestingly, in the toxic column, a lot of the comments use an abundance of capital letters. Furthermore, comments deemed more toxic seem to have more misspelling, slang, and punctuation. Of course, this type of identification is for the most obvious cases. However, this leads me to wonder if we can apply an algorithm to identify underlying trends.   
#   
# I almost glossed over one thing: The goal of this project is to rank the comments from most toxic to least toxic. As explained in the project description, it is very difficult to rank a list of comments entirely. Instead, it is a lot easier to judge two comments and rank which one is toxic. I should keep this in mind when I'm building my future algorithm. Instead of loading the entire dataset into my model, it might be better to use a sliding window method that takes in two comments and spits out which one is considered more toxic. This leads me to think that a binary classifier might be particularly useful. However, how would we identify the unlabeled features? Hmmm.... How much natural language processing is actually necessary here?
#   
# On a side note, I think it might be interesting to see the amount of overlap between the two classes (more toxic vs less toxic)
# I find it funny that there's more toxic comments than not toxic comments. 

# In[11]:


common_df = {}
un_not_toxic_df = {}


count = 0
ncount = 0

for i in range(len(not_toxic_df.text)):
    ntword = not_toxic_df.text[i]
    ntcount = not_toxic_df['count'][i]

    if ntword in list(toxic_df.text):
        t_idx = (toxic_df.index[toxic_df.text == ntword].tolist())[0]
        t_count = toxic_df['count'][t_idx]
        tword = toxic_df['text'][t_idx]
        common_df[count] = [ntword, ntcount, t_count]
        count += 1
        
    else:
        un_not_toxic_df[ncount] = [ntword, ntcount]
        ncount += 1
        
common_df = pd.DataFrame.from_dict(common_df, orient = 'index', columns = ['text', 'ntcount', 'tcount'])
un_not_toxic_df = pd.DataFrame.from_dict(un_not_toxic_df, orient = 'index', columns = ['text', 'count']).sort_values(by = ['count'], ascending = False)
un_not_toxic_df = un_not_toxic_df.reset_index(drop = True)


# In[12]:


count = 0
un_toxic_df = {}

for i in range(len(toxic_df)):
    tword = toxic_df.text[i]
    if tword not in list(common_df.text):
        tcount = toxic_df['count'][i]
        un_toxic_df[count] = [tword, tcount]
        count += 1

un_toxic_df = pd.DataFrame.from_dict(un_toxic_df, orient = 'index', columns = ['text', 'count']).sort_values(by = ['count'], ascending = False)
un_toxic_df = un_toxic_df.reset_index(drop = True)


# In[13]:


common_df = common_df.sort_values(by = ['tcount', 'ntcount'], ascending = False)
print(common_df[0:10])
print("\nshared comments count: ", len(common_df))
print("toxic comments total: ", len(toxic_df))
print("less toxic comments total: ", len(not_toxic_df))
print("% of overlap for toxic: ", (round(len(common_df)/len(toxic_df),3)*100))
print("% of overlap for less toxic: ", (round(len(common_df)/len(not_toxic_df),3)*100))


# This is interesting... There is a huge overlap between our more toxic and less toxic comments. Furthermore, the percentage of overlap is similar in both cases, with 76.7% overlap for toxic and 77.7% for less toxic which makese sense given that both datasets are similar sizes. This also tells me that close to 25% of each dataset is uniquely toxic and not toxic, which is also worth exploring

# In[14]:


print("Uniquely Toxic Comments, length: ", len(un_toxic_df))
un_toxic_df[0:10]


# In[15]:


print("Uniquely Not Toxic Comments, length: ", len(un_not_toxic_df))
un_not_toxic_df[0:10]


# I believe looking at this new data supports what I was saying previously, which is that all caps comments are a great way to identify toxic comments. However, as I said before, this will only help to discern the most obvious cases. This remains true with regarding less toxic comments and proper grammar. Later, as I build my model, I should keep in mind ways to check for proper spelling and grammar. 
# 
# If I think about this in larger scenario, should I create my own scoring method. This might help with unsupervised learning. For example, I could add points based off of the ratio of capital words to total words, which will push all my capital words to the most toxic part of my list...
# 
# 
# ### Update 1/12/2022
# - If I had to think about what type of algorithm I should use, it would be something like semi-supervised or unsupervised. However, now I need to think about how I would like to preprocess my data... Furthermore, I should be using my validation as exactly that; Validation data.... Maybe I could take a part of my validation data and use that as a training set and validate that afterwards? I'll need to double check on that....
#      
# 
# 
# ### Let's discuss potential algorithms to use: 
# 
# ###### Naive Bayes / SVM
# - Supervised Learning method. It'll require me to know what features I'm looking for. I discussed potential features above, which include the amount of misspelled words, typos, swears, capitalized letters, and punctuation.
#      - One way to prove the validity of this statement is to create some histograms. 
#           - For example, capital letters: Each bin will contain different ratios of total capitalized letters to total amount of words. The Y axis is the frequency of counts per toxic or not toxic.       
#             
#             
# - However, my particular issue with Naive Baye's in concept is that this only takes into features that I can tell right off the bat. What about features that I don't know about? 
# 
# ##### Unsupervised Learning Methods (Deep Learning)
# - The biggest benefit here is the ability to extract unknown features. Since this is considered an NLP project, CNN and RNN models are considered, although I'm fairly certain that CNN is used to identify images or handwritten digits. This leads me to think that creating an RNN model through Keras might be the best foot going forward, but I will need to read more about this to get a better understanding. Plus, I believe this area will require more preproccessing of my data, which I'm still not sure how to go about it
# 
# 
# ##### Other Notes: 
# I was scouring the kaggle sites, looking at work other people have done. I noticed that a good portion of them use the Ridge Regression model, which I don't yet understand why...
# - This is a model tuning method, which uses L2 regularization.... I'll need to figure how I might utilize this later. 
# 

# ## Naive Baye's
# ### Building the Histogram
# First, let's think about separating our data. We should have a training set, validation set, and testing set, which we will take from our validiation.csv. 
#   
#   
# In the training set, we will build our histograms and get some probabilities. From what I've observed in other people's code; they try to clean the data by removing special characters. As I stated previously, I don't think I agree with this. A very 'naive' way to go about this is to build the model off text only. 
# 
#   
# p(CapitalizedWords|Toxic) * p(mispelledwords|Toxic) * p(punctuation|toxic) * p(swears|toxic)  
# p(CapitalizedWords|not toxic) * p(mispelledwords|not toxic) * p(punctuation|not toxic) * p(swears|not toxic)  
#   - But wait! These features seem to mainly lean towards toxic. What about features that are noticable for less toxic?  
#     - Honestly I don't see anything that is... distinguishable... It's basically proper punctuation and properly spelled words.
#     - Potential issues:
#         - We shouldn't forget, that we also don't know about ranking... How will we score it?
#         - Remember earlier, when I was talking about how this might only detect the most obvious cases? What about cases that aren't so obvious, like a rude passive aggressive comment that displays proper grammar? This might just be room for error, but still..... You know what whatever! Let's build a beta version first
#   
#   
# Preprocessing my data:   
# Luckily for me, I started preprocessing my data, which are toxic_df and not_toxic_df. As stated previously, these datasets are pretty balanced. Actually, as I think about it, I can just pull random datapoints from my unique datasets and my common datasets and then appending them, given that the distribution is pretty balanced.   
# It might be worth interesting to treat each punctuation (specifically exclamation points) as entire words. 
# 
# 
# #### There might also be a dataset that could be helpful, which is a kaggle provided dataset that rated comments based off of severity.
# - Another naive baye's classifier that I saw utilized this to train. 

# In[16]:


temp = common_df.sort_values(by = ['ntcount', 'tcount'], ascending = False);
temp = temp.reset_index(drop = True)
print(temp.text)
print(temp)


# In[17]:


import numpy as np

train_t, valid_t, test_t = np.split(un_toxic_df.sample(frac = 1, random_state = 42), [int(.6*len(un_toxic_df)), int(.8*len(un_toxic_df))])
train_nt, valid_nt, test_nt = np.split(un_not_toxic_df.sample(frac = 1, random_state = 42), [int(.6*len(un_not_toxic_df)), int(.8*len(un_not_toxic_df))])
train_c, valid_c, test_c = np.split(common_df.sample(frac = 1, random_state = 42), [int(.6*len(common_df)), int(.8*len(common_df))])

