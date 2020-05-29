import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import os
import sys

#change the path
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

def gbdt(args):
    if len(args)==2:
        file_name=args[0]
        table_name=args[1]
        
        test = pd.read_excel(file_name, sheet_name=table_name, na_values='n/a')
        ######### reading data 
        test_x = test[['duration', 'size', 'bite_rate', 'width', 'height', 'r_frame_rate', 'bitrate', 'resolution', 'framerate','mpeg4','vp9','hevc']]
        test_y = test[['percentage']]

        #Standardization of input
        scaler = StandardScaler()
        scaler.fit(test_x)
        test_x = scaler.transform(test_x)

        reg = joblib.load('GBDT.m') #upload the model
        print(reg)

        y_pred = reg.predict(test_x)
        y_pred = y_pred.reshape(-1,1)
        data = pd.DataFrame(y_pred)
        data.to_excel("prediction.xlsx",sheet_name='prediction result',header=False,index=False)
        return y_pred 
    
    elif len(args)==12:
        duration = args[0]
        size = args[1]
        bite_rate = args[2]
        width = args[3]
        height = args[4]
        r_frame_rate = args[5]
        bitrate = args[6]
        resolution = args[7]
        framerate = args[8]
        mpeg4 = args[9]
        vp9 = args[10]
        hevc = args[11]
        
        test_x = [[duration, size, bite_rate, width, height, r_frame_rate, bitrate, resolution, framerate, mpeg4, vp9, hevc]]
        
        scaler = StandardScaler()
        scaler.fit(test_x)
        test_x = scaler.transform(test_x)

        reg = joblib.load('GBDT.m') #upload the model
        print(reg)

        y_pred = reg.predict(test_x)
        return y_pred 
    
    else:
        print("Input error, please check")

pred = gbdt(sys.argv[1:]) 
#print(pred)






    



