import pandas as pd
import numpy as np
import os
import xlrd

def ZC_Data_extractor(path):
# path to the excel file
    os.chdir(path)

    while True :
        try :
            input_file = xlrd.open_workbook('Swaption Pricer.xls')
        except :
            print("input file is not in the directory")
        break

    IR_Curve = input_file.sheet_by_name('IR Curve')

    time = []
    j = 0
    while True:
        try:
            time.append(float(IR_Curve.cell(7 + j, 7).value))
            j = j + 1
        except:
            break

    Rate = []
    j = 0
    while True:
        try:
            Rate.append(float(IR_Curve.cell(7 + j, 8).value))
            j = j + 1
        except:
            break

    # create dataframe for ZC_Curve
    return pd.DataFrame(data=np.array([time, Rate]).T, columns=['Time', 'Rate'])
