#file handling functions
def info():
    print('''

6 functions;

    3 for lists;
        lwrite()
        lappend()
        lread()
        
    3 for Dictionaries;
        dwrite()
        dappend()
        dread()
''')


#LISTS

#write
'''
lwrite(headers,data,'destination')'''

def lwrite(head,data,dest):
    import csv
    f = open(dest,'w',newline='')
    w = csv.writer(f)
    w.writerow(head)
    w.writerows(data)
    f.close()

#append
'''
'lappend(data,'destination')'''

def lappend(data,dest):
    import csv
    f = open(dest,'a',newline='')
    w = csv.writer(f)
    w.writerows(data)
    f.close()
 
#read
'''
lread('destination')

Output: headers, data'''

def lread(dest):
    import csv
    global head, data
    f = open(dest,'r',newline='')
    r = csv.reader(f)
    head = []
    data = []
    head = next(r)
    for i in r:
        data.append(i)
    # print('Headers;')
    # print(head)
    # print('Data;')
    # for i in data:
        # print(i)
    f.close()
    return head, data

#DICTIONARIES

#write
'''
dwrite(headers,data,'destination')'''

def dwrite(head,data,dest):
    import csv
    f = open(dest,'w',newline='')
    dw = csv.DictWriter(f,fieldnames=head)
    dw.writeheader()
    dw.writerows(data)
    f.close()

#append
'''
dappend(headers,data,'destination')'''

def dappend(head,data,dest):
    import csv
    f = open(dest,'a',newline='')
    dw = csv.DictWriter(f,fieldnames=head)
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