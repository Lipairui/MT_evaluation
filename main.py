import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import re

# Load model from tensorflow hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/2")
# Load model from local
# embed = hub.KerasLayer('../../Sentence_embeddings/Universal_sentence_encoder/model/')

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
        
 def evaluate(sentences1,sentences2,trans_type):
    '''
    Input:
        sentences1 [list]: source sentences in translations.
        sentences2 [list]: target sentences in translations.
        trans_type [str]: language type of translations. E.g. "en-zh" stands for "English to Chinese".
    Output:
        results [list]: score of each pair of translations. The higher the score is, the better the translation is. Score ranges in [0,1].
    '''
    assert len(sentences1)==len(sentences2)
    psentences1 = [text_preprocess(sentence1,lang=trans_type[:2]) for sentence1 in sentences1]
    psentences2 = [text_preprocess(sentence2,lang=trans_type[-2:]) for sentence2 in sentences2]
    embeddings1 = embed(psentences1)["outputs"]
    embeddings2 = embed(psentences2)["outputs"]
    similarity_matrix = np.inner(embeddings1,embeddings2)
    results = []
    for i in range(len(sentences1)):
        sim = similarity_matrix[i][i]
        results.append(sim)
    return results

if __name__ == "__main__":
    sentences1 = ["I love you!","I want to eat fried rice tonight.","I am glad to hear that"]
    sentences2 = ["我爱你","我今晚想吃炸薯条","听到这件事我很难过"]
    trans_type = "en_zh"
    scores = evaluate(sentences1,sentences2,trans_type)
    for i in range(len(sentences1)):
        print('Source: ',sentences1[i])
        print('Target: ',sentences2[i])
        print('Score: ',scores[i])
