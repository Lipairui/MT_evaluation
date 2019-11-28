import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import re

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/2")

def text_preprocess(text,lang):
    if lang=='en':
        # TEXT CLENAING
        TEXT_CLEANING_RE = "[^A-Za-z0-9]"
        # Remove link,user and special characters
        ptext = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower())
        # Remove extra whitespace
        ptext = re.sub('[\s]{2,}',' ',ptext).strip()
        return ptext
    elif lang=='zh':
        # TEXT CLENAING
        TEXT_CLEANING_RE = "[^\u4E00-\u9FFF0-9]"
        # Remove link,user and special characters
        ptext = re.sub(TEXT_CLEANING_RE, ' ', text)
        # Remove extra whitespace
        ptext = re.sub('[\s]{2,}',' ',ptext).strip()
        return ptext
    else:
        return text
        
 def cal_score(sentences1,sentences2):
    embeddings1 = embed(sentences1)["outputs"]
    embeddings2 = embed(sentences2)["outputs"]
    similarity_matrix = np.inner(embeddings1,embeddings2)
    results = []
    for i in range(len(sentences1)):
        sim = similarity_matrix[i][i]
        results.append(sim)
    return results
