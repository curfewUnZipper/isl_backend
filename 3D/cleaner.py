import os
import csvLibrary as cl
# left is 0, right is 1 in excel sheets


#storing rlb-012 data in a list of list
csv_name_list=[]

for entry in os.scandir("./trainingData"):
    csv_name_list.append(entry.path)

with open("LRB-012.txt", "r") as file:
    content = file.read()
# print(content)
content= content.split("\n")
newContent = {}
for i in content:
    if f"./trainingData\\{i[0]}.csv" in csv_name_list:
        newContent[i[0]] = i[-1]
# print(newContent,'\n\n',list(newContent.items()),'\n\n')
# print(newContent["A"])

################### Code for cleaning ###################

# access Alphabets using newContent["A"]
# print(list(past[0].items())[0:21]) # [0:21] or [21:42]

#gets teh name of current letter, and then get which hand to be used then delete the other hand
#~ naah <-- then again go through the data which is required but is empty/lacking and delete that


logTrainingData = {} #to store the no. of trainingData per alphabet
for i in list(newContent.keys()):
    currentFile = cl.dread(f"./trainingData/{i}.csv")
    print(f"Letter: {i}, Hand: {newContent[i]}")
    if int(newContent[i])!=2: #left hand coordinates
        for j in range(len(currentFile)):
            for k,l in list(currentFile[j].items())[abs(int(newContent[i])-1)*21 : abs(int(newContent[i])-1)*21+21]:
                currentFile[j][k]=''
    
    else:
        print("Running 2 hand cleaning fn")
        """
       iterate through a list of dicts, obtain dict.values, then do .find('') if yes, delete the dict 
       used variable count: i
        """
        j = 0
        frames=len(currentFile)
        while j<frames:
            if '' in currentFile[j].values():
                currentFile.pop(j)
                j-=1
                frames-=1
            j+=1
    
    
    
    #removing empty lists from output
    n=0
    while (n<len(currentFile)):
        if currentFile[n] != {
  "0.00": "",
  "0.01": "",
  "0.02": "",
  "0.03": "",
  "0.04": "",
  "0.05": "",
  "0.06": "",
  "0.07": "",
  "0.08": "",
  "0.09": "",
  "0.10": "",
  "0.11": "",
  "0.12": "",
  "0.13": "",
  "0.14": "",
  "0.15": "",
  "0.16": "",
  "0.17": "",
  "0.18": "",
  "0.19": "",
  "0.20": "",
  "1.00": "",
  "1.01": "",
  "1.02": "",
  "1.03": "",
  "1.04": "",
  "1.05": "",
  "1.06": "",
  "1.07": "",
  "1.08": "",
  "1.09": "",
  "1.10": "",
  "1.11": "",
  "1.12": "",
  "1.13": "",
  "1.14": "",
  "1.15": "",
  "1.16": "",
  "1.17": "",
  "1.18": "",
  "1.19": "",
  "1.20": ""
        }: #these are empty coord full dictionary {0.01:'', 0.02:''.....}
            # print("\n\nOUTPUT: ",currentFile[i])
            n+=1
        else:
            currentFile.pop(n)
            # print(str(i)+"/"+str(len(currentFile)))
     
    #preparing upload 
    from decimal import Decimal
    headers = []
    for m in range(2):
        for j in range(21):
            # print(j)
            headers.append(str(round(m+Decimal(j/100),2)))
    # print(headers)
    # print(currentFile[0])

    #store the no. of training data per alphabet in a file as well:
    logTrainingData[i]=len(currentFile)

    cl.dwrite(headers,currentFile,f"./trainingData/{i}.csv")
with open("TrainingLog.txt","w") as file:
    file.write("TRAINING DATA PER ALPHABET:\n\n")
    for i,j in logTrainingData.items():
        file.write(f"{i} - {j}\n")