# Machine Translation Evaluation
Automatic evaluate Machine Translation(MT) results based on Google's Universal Sentence Encoder. Evaluate translation only based on source and target sentences. Reference sentences are not necessary to be provided! 

## Example usage
    from main import evaluate
    sentences1 = ["I love you!", "Nice to meet you", "I want to eat fried rice tonight.", "I am glad to hear that"]
    sentences2 = ["我爱你", "很高兴见到你！", "我今晚想吃炸薯条", "听到这件事我很难过"]
    trans_type = "en_zh"
    scores = evaluate(sentences1,sentences2,trans_type)
    for i in range(len(sentences1)):
        print('Source: ',sentences1[i])
        print('Target: ',sentences2[i])
        print('Score: ',scores[i])
Result:

    Source:  I love you!      
    Target:  我爱你      
    Score:  0.9733935      
    Source:  Nice to meet you         
    Target:  很高兴见到你！         
    Score:  0.81520224         
    Source:  I want to eat fried rice tonight.         
    Target:  我今晚想吃炸薯条       
    Score:  0.57846886        
    Source:  I am glad to hear that       
    Target:  听到这件事我很难过       
    Score:  0.5725441

## Reference
https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/2
