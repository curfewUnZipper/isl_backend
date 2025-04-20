#write the reader code and then open it as r+ mode in translator.py
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import wordninja
def sentenceMaker():
    readSentenceFile = open(list(os.scandir("./signLog"))[-1].path,"r")
    text = readSentenceFile.read()    #use split- ' ' and '\n'
    readSentenceFile.close()
    print("File:",text)
    # nlp model to identify - this is done in steps
    # 1. Ninja
    words = wordninja.split(text)
    corrected_text = ' '.join(words)
    print("NinjaWords:", corrected_text)

    # 2. Transformers - BERT
    if len(corrected_text)>5:
        tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
        model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")
        input_text = "fix grammar: " + corrected_text
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Transformers:",corrected_text)
    return corrected_text
if __name__ == "__main__":
    print(sentenceMaker())