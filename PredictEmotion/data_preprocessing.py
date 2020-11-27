import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data
ra2_1 = pd.read_pickle("./data/pickle/201012_ra2_1.pkl")
ra3_1 = pd.read_pickle("./data/pickle/201012_ra3_1.pkl")
ra4_1 = pd.read_pickle("./data/pickle/201012_ra4_1.pkl")
ra5_0 = pd.read_pickle("./data/pickle/201012_ra5_0.pkl")

def preprocessing(df, concentrate):
    #remove 0 values
    data = df[(df!=0).all(1)]

    #remove miss detected value with position
    bins = np.linspace(data.min(axis=0)[7], data.max(axis=0)[7], 10)
    hist, _ = np.histogram(data.X_pos, bins)
    data = data.drop(data[(data.X_pos < max(bins[np.argmax(hist)-1], bins[0])) | (data.X_pos > min(bins[np.argmax(hist)+2], bins[-1]))].index)
    
    bins = np.linspace(data.min(axis=0)[8], data.max(axis=0)[8], 10)
    hist, _ = np.histogram(data.Y_pos, bins)
    data = data.drop(data[ (data.Y_pos < max(bins[np.argmax(hist)-1], bins[0])) | (data.Y_pos > min(bins[np.argmax(hist)+2], bins[-1])) ].index)

    #conjugate label
    label = [concentrate] * len(data)
    data = data.drop(['X_pos','Y_pos','scale'], axis=1)
    data.insert(7, 'label', label)
    
    return data

#append data
ra2 = preprocessing(ra2_1, 1)
ra3 = preprocessing(ra3_1, 1)
ra4 = preprocessing(ra4_1, 1)
ra5 = preprocessing(ra5_0, 0)

ra = pd.concat([ra2, ra3, ra4, ra5])
ra = ra.reset_index(drop=True)

ra_shuffled = ra.sample(frac=1).reset_index(drop=True)
print(ra_shuffled)

#save data in pickle
ra_shuffled.to_pickle("./data/analysis/201012/data_prepared.pkl")
