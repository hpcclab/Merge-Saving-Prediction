import os
import re
import xlwt

file_name = '_r_base'
file = []
input_name = []
operation = []
time_real = []
time_user = []
time_sys = []
operator = ['r10','r15','r20','r30','r40']

#Listing the file name with a keyword in the documents, every time change the configuration filename, adjusting the function keyword
def search(path, keyword):
    content = os.listdir(path)
    for each in content:
        each_path = path + os.sep + each
        if keyword in each:
            #print(each_path)
            file.append(each_path)
            if os.path.isdir(each_path):
                search(each_path, keyword)

#search(os.getcwd(), 'sm')
#print(file)

def content_search(path, keyword1, keyword2, keyword3, keyword4):
    with open(path,'r') as file:
        for line in file:
            if keyword1 in line:
                temp1 = line.rsplit(None)
                time_real.append(temp1[1])
            if keyword2 in line:
                temp2 = line.rsplit(None)
                time_user.append(temp2[1])
            if keyword3 in line:
                temp3 = line.rsplit(None)
                time_sys.append(temp3[1])
            if keyword4 in line:
                temp4 = line.rsplit(None)
                input_name.append(temp4[1])
            '''
            if keyword5 in line:
                temp5 = line.rsplit(None)
                operation.append(temp5[1:])
            '''
search(os.getcwd(), file_name)
print(file)
workbook = xlwt.Workbook()

n = 0
for op in operator:
    #file_table = path.rsplit("/")
    sheet = workbook.add_sheet(f"{op}")
    sheet.write(0, 0, 'input')
    # sheet.write(0, 1, 'operation')
    sheet.write(0, 1, 'real time')
    sheet.write(0, 2, 'user time')
    sheet.write(0, 3, 'sys time')
    sheet.write(0, 4, f'{op}')
    i = 1
    for path in file:
        input_name = []
        #operation = []
        time_real = []
        time_user = []
        time_sys = []
        #content_search(path, 'real', 'user', 'sys', 'input', 'operation')

        #print(len(time_user),len(time_real),len(time_sys))
        content_search(path, 'real', 'user', 'sys', 'input:')
        cou = int(len(input_name)/5)

        #reset following parameters
        for co in range(cou):
            #name = re.findall(r"\d+",path)
            #str = "".join(name)
            #sheet.write(0, i, str)
            #a = len(input_name)
            #b = len(time_sys)
            #if a == b:
            #    None
            #else:
            #    print(path)
            sheet.write(i, 0, input_name[co*5 + n])
            #print(input_name,co)
            #sheet.write(i, 1, operation[co])
            #try:
            sheet.write(i, 1, time_real[co*5 + n])
            sheet.write(i, 2, time_user[co*5 + n])
            sheet.write(i, 3, time_sys[co*5 + n])
            #except IOError:
            #    print("worry happen in:" + path)
            #else:
            #    print("success")
            i = i + 1
    n = n + 1
workbook.save(f'{file_name}.xls')
'''

'''