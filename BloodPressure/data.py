import csv
import numpy as np

data = []
i = 0
with open('ppdata/3000393.csv', 'r', encoding='utf8') as f:
    lines=csv.DictReader(f)
    for line in lines:
        if i>0:         
            pleth = line["'PLETH'"]
            ii = line["'II'"]
            abp = line["'ABP'"]
            avr = line["'AVR'"]
            if pleth == '-' or ii == '-' or abp == '-' or avr =='-':
                continue    
            time = float(line["'Elapsed time'"])
            data.append([time, float(pleth), float(avr), float(ii), float(abp)])
        i+=1   
np.save('3000393.npy', data)



        