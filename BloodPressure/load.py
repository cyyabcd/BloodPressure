import numpy as np

data = np.load('data/3000393.npy')
t = 0
mdata = []
for i in range(data.shape[0]-1):
    if abs(data[i][0] - data[i+1][0])<0.0081:
        mdata.append(data[i])
    else:
        np.save('data/3000393/%d.npy'%(t),mdata)
        mdata = []
        t +=1
np.save('data/3000393/%d.npy'%(t),mdata)


