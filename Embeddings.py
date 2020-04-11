import numpy as np
import bcolz
import pickle
from nltk.corpus import stopwords
import os.path
import re


def construct_embedding_matrix(dim):
    """
    args:
        dim (int): the dimension of the embedding vectors
        
    returns:
        matrix of shape (nterms,dim) containing the embedding vector of each term of the vocabulary
    """
    words=['UNKNOWN']
    idx=1
    word2id={'UNKNOWN':0}
    #embeddings=np.random.uniform(-1,1,dim)
    embeddings = bcolz.carray(np.random.uniform(-1,1,dim), rootdir=f'./Data/glove.6B/6B.'+str(dim)+'.dat', mode='w')
    stoplist=stopwords.words("english")

    with open(f'./Data/glove.6B/glove.6B.'+str(dim)+'d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            if (word not in stoplist and len(word)>2):
                words.append(word)
                word2id[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                embeddings.append(vect)
            #embeddings=np.vstack((embeddings,vect))
    embeddings = bcolz.carray(embeddings.reshape((-1, dim)), rootdir=f'./Data/glove.6B/6B.'+str(dim)+'.dat', mode='w')
    embeddings.flush()
    pickle.dump(words, open(f'./Data/glove.6B/6B.'+str(dim)+'_words.pkl', 'wb'))
    pickle.dump(word2id, open(f'./Data/glove.6B/6B.'+str(dim)+'_idx.pkl', 'wb'))
    
    
def extract_aol_queries(path):
    queries=[]
    for f in os.listdir(path):
        lignes=open(path+f).readlines()
        f_queries=np.unique([re.split('\t',q)[0]for q in list(filter(lambda x: ('http' not in x and x!='\n'),lignes))])
        for q in f_queries:
            queries.append(clean(q))
            
    #queries=bcolz.carray(queries[1:], rootdir=f'./Data/AOL_queries.dat', mode='w')
    #queries.flush()
    return queries
    
    
def clean(to_clean):
    text_cleaned = re.sub("<!-- (.+?) -->", "", ''.join(to_clean))
    text_cleaned = re.sub("`", "", text_cleaned)
    text_cleaned = re.sub("'", "", text_cleaned)
    text_cleaned = re.sub("\"", "", text_cleaned)
    text_cleaned = re.sub("'", "", text_cleaned)
    text_cleaned = re.sub("  ", "", text_cleaned)

    text_cleaned = re.sub("\\udcc6","",text_cleaned)
    text_cleaned = re.sub("\\udce6", "", text_cleaned)
    text_cleaned = re.sub("\\udcc5", "", text_cleaned)
    text_cleaned = re.sub("\\udceb", "", text_cleaned)
    text_cleaned = re.sub("\\udce3", "", text_cleaned)
    text_cleaned = re.sub("\\udce3", "", text_cleaned)
    text_cleaned = re.sub("\\udcf8", "", text_cleaned)
    text_cleaned = re.sub("\\udcec", "", text_cleaned)

    return text_cleaned    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
