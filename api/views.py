from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
from django.conf import settings
from django.http import HttpResponse


import mediapipe as mp
import cv2
import time
from datetime import datetime, timedelta
import os
import sys
import numpy as np
from decimal import Decimal
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import wordninja



@csrf_exempt

# def double_dict(request):
#     try:
#         img_data = request.data  # <-- keep it as binary
#         timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')
#         filename = os.path.join(save_dir, f"{timestamp}.jpg")

#         with open(filename, 'wb') as f:
#             f.write(img_data)

#         return jsonify({"message": "Frame received"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500




def double_dict(request):
    if request.method == 'POST':
        try:
            data = request.body
            #enter the changed code here
            load_saved_artifacts()
            # return HttpResponse("loaded")
            finalOut = createResultCSV(data)      
            
            return HttpResponse(finalOut)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Only POST method is allowed'}, status=405)


def createResultCSV(feedImg):
    headers=[]

#     cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    sentenceFile = open("./signLog/"+time.strftime("%Y-%m-%d_%H-%M-%S") + ".txt","a+")
    sentenceFile.seek(0)
# readSentenceFile = open("./signLog/"+time.strftime("%Y-%m-%d_%H-%M-%S") + ".txt","r")

#     count={}
#     prevAns=""
#     ans=""
#     sentence = ""
#     actionTime = datetime.now()
#     ansTime = datetime.now()
#     #run for unlimited time
#     while True:            
#         pTime = 0
#         cTime = 0
#         timerTime = time.time()
#         output = []
#         headers=[]
    partOutput=[]

        
    
#         # Run for 03 seconds
#         while time.time()<(timerTime+0.5):

            # success, img= cap.read()
    # Convert bytes into numpy array
    nparr = np.frombuffer(feedImg, np.uint8)

    # Decode into OpenCV image (BGR format)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Image decoding failed!")
    img=cv2.flip(img,1)
    # cv2.imshow("showing live feed",success)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    halfList1 = []
    halfList2 = []
    qtrList = []

    if (results.multi_hand_landmarks):
        
        # print("next frame") 

        #interesting detail:
        # print(results.multi_handedness[1]) #gives left, but its index property = 0
        # print(results.multi_handedness[0]) #gives right, but its index property = 1

        # print(results.multi_handedness)
        for handSide in results.multi_handedness:
            # print(handSide.classification[0].index)
            halfList1.append(handSide.classification[0].index)


        for handLms in results.multi_hand_landmarks:
            # print(handLms.landmark)
            qtrList=[]
            #this area has both hands seperately
            for id,lm in enumerate(handLms.landmark):
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                #print(id,lm) #this gives position (as a fraction of img) for each pt
                height,width,channel = img.shape #finding img parameters to multiply to lm to get coordinate 
                cx,cy = int(lm.x*width), int(lm.y*height)
                qtrList.append([id,cx,cy])
                # if id==4 or id==3:
                    # print(id, cx,cy)
                    # qtrList.append([id,cx,cy])

            halfList2.append(qtrList)
        #this area is after a frame
        # print(halfList1)
        # print(halfList2)
        # print("\n\n\n")
        partOutput.append(dict(zip(halfList1,halfList2)))   
        # print(partOutput)

    """
    this is the test phrase (returns A) and this is the accepted format for call:
{"1": "[[0, 465, 459], [1, 423, 438], [2, 395, 406], [3, 379, 381], [4, 369, 360], [5, 424, 357], [6, 379, 374], [7, 377, 403], [8, 387, 416], [9, 442, 365], [10, 392, 395], [11, 393, 422], [12, 403, 433], [13, 456, 381], [14, 407, 410], [15, 410, 435], [16, 420, 445], [17, 467, 402], [18, 426, 423], [19, 426, 444], [20, 435, 452]]", "0": "[[0, 209, 445], [1, 246, 435], [2, 272, 404], [3, 285, 376], [4, 297, 352], [5, 239, 360], [6, 252, 401], [7, 252, 418], [8, 249, 420], [9, 210, 366], [10, 228, 417], [11, 231, 432], [12, 229, 435], [13, 185, 381], [14, 200, 427], [15, 207, 441], [16, 206, 443], [17, 163, 400], [18, 175, 436], [19, 182, 447], [20, 182, 448]]"}

"""
    ansTime=0
    prevAns=0
    output=[]
    
    # partOutput = input("Enter the dict: ")
    
    # import json
    # converted = {int(k): json.loads(v) for k, v in partOutput.items()}
    # partOutput=[converted]
    # output.append(partOutput)
    # print(partOutput)

    # #displayingFP
    # cTime = time.time()
    # fps = 1/(cTime-pTime)
    # pTime= cTime
    # cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)

    # ###################################################   PUT ALPHABET+SENTENCE ON SCREEN   ##############################
    # cv2.putText(img,str(sentence),(65,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),3)
    # if len(prevAns)>0:
    #     cv2.putText(img,str(ans),(300,450),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
    #     ansTime = datetime.now()
    # cv2.imshow("Image",img)
    # cv2.waitKey(1)
    # # time.sleep(2)




    ########################################### PUTTING OUTPUT IN CSV
    #creating headers list for csv            
    for i in range(2):
        for j in range(21):
            # print(j)
            headers.append(round(i+Decimal(j/100),2))

    #removing empty lists from output
    i=0
    while (i<len(output)):
        if output[i] != []:
            # print("\n\nOUTPUT: ",output[i])
            i+=1
        else:
            output.pop(i)
            # print(str(i)+"/"+str(len(output)))

    finalOutput = []
    #converting output to the perfect dictionary wali list
    for i in range(len(partOutput)):
        # print(output[i].keys())
        tempDict={}
        for j in partOutput[i].keys():
            for k in partOutput[i][j]:
                # print(round(j+Decimal(k[0]/100),2),k[1:])
                tempDict[round(j+Decimal(k[0]/100),2)] = k[1:]

        #creating dict of one frame and appending it here            
        finalOutput.append(tempDict)
    # print(finalOutput)
    #printing headers, output        
    # print("HEADERS:",headers)
    # print("OUTPUT:",finalOutput)
    # print("\n\nFrames Captured =",len(finalOutput))
    # left is 0, right is 1
    dwrite(headers,finalOutput,"./currentOutput.csv")
        







    #########################################another section starts here


    X=[]
    y=[]
    path = "./currentOutput.csv"
    # print(path[15])
    csvData = dread(path)

    for rowEntry in csvData:
        # print(rowEntry)
        tempRow = []
        for handPoint, coords in rowEntry.items():
            if coords != '':
                coords = list(coords[1:-1].split(","))
                # print(float(handPoint), int(coords[0]),int(coords[1]))
                tempRow.extend([float(handPoint), int(coords[0]),int(coords[1])])
            else:
                tempRow.extend([float(handPoint),-600,-600])
        X.append(tempRow)
        final = np.array(tempRow).reshape(1,126) #1 row and 3*42 entries
        # print('X:',X,"\ny:",y)
    # tempRow
    # final
        checkDict = {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 
            'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 
            'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 
            'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 
            'Y': 24, 'Z': 25
        }
        # print(__model.predict_proba(final))
        output = __model.predict(final)[0]
        # print("prediction:",output,":",float((__model.predict_proba(final)*100)[0,checkDict[str(output)]]))
        # print(output)
        return output
        # if float((__model.predict_proba(final)*100)[0,checkDict[str(output)]])>95:
        #     y.append(__model.predict(final)[0])
        #     ans = str(y[-1])
            
            # print(float((__model.predict_proba(final)*100)[0,checkDict[str(output)]]))
        # print(__model.predict(final),":",float((__model.predict_proba(final)*100)[0,0])>50,type((__model.predict_proba(final)*100)[0,5]>50))

    # print("X[0]: ",X[0])
    # print("\n\nY: ",y,len(y))
    # print("prev:",prevAns,"now:",ans)
    # if ansTime>(datetime.now()-timedelta(seconds=2)):
    #     ans=""
    # if prevAns != ans:
    #     # print(prevAns,ans,"here")
    #     # print(datetime.now())
    #     prevAns = ans
    #     # print("type of datetime subtraciton:",type(datetime.now()-actionTime))
    #     actionTime = datetime.now()
    #     sentenceFile.write(str(prevAns))
    #     sentenceFile.flush()
    #     sentence = sentenceMaker()      
    #     ####################################### TTS - Text To Speech ###################################
        # engine.say(sentence)  
        # engine.runAndWait()
        
            # else:
            # 












######################################################### Artifacts loading
import joblib
import json

__class_name_to_number = {}
__class_number_to_name = {}

__model = None


def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    # print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open(os.path.join(settings.BASE_DIR, 'artifacts', 'class_dictionary.json'), "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open(os.path.join(settings.BASE_DIR, 'artifacts', 'saved_model.pkl'), 'rb') as f:
            __model = joblib.load(f)
    # print("Artifacts Loaded")


##################################### SENTENCE MAKER #################################
def sentenceMaker():
    readSentenceFile = open(list(os.scandir("./signLog"))[-1].path,"r")
    text = readSentenceFile.read()    #use split- ' ' and '\n'
    readSentenceFile.close()
    # print("File:",text)
    # nlp model to identify - this is done in steps
    # 1. Ninja
    words = wordninja.split(text)
    corrected_text = ' '.join(words)
    # print("NinjaWords:", corrected_text)

    # 2. Transformers - BERT
    # if len(corrected_text)>5:
    #     tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
    #     model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")
    #     input_text = "fix grammar: " + corrected_text
    #     inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    #     outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    #     corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     print("Transformers:",corrected_text)
    return corrected_text





def dwrite(head,data,dest):
    import csv
    f = open(dest,'w',newline='')
    dw = csv.DictWriter(f,fieldnames=head)
    dw.writeheader()
    dw.writerows(data)
    f.close()

#read
'''
dread('destination')

Output: data'''

def dread(dest):
    import csv
    global data
    f = open(dest,'r',newline='')
    dr = csv.DictReader(f)
    data = list()    
    for i in dr:        
        data.append(dict(i))
    f.close()
    return data


