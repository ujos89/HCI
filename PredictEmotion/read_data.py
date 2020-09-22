#read data from pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_pickle("./data/data_200922.pkl")
df2 = pd.read_pickle("./data/data_200922_2.pkl")
print(df1)
#print(df2)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]
bin_ = np.linspace(0,1,41)
data_hist = pd.DataFrame(columns=bin_[:-1])

for i in range(len(EMOTIONS)):
    hist, bins = np.histogram(df1[EMOTIONS[i]], bins=bin_)
    hist_temp = {}
    for j in range(len(bin_)-1):
        hist_temp[data_hist.columns[j]] = hist[j]
    data_hist = data_hist.append(hist_temp, ignore_index=True)

data_hist = data_hist.T
data_hist.columns = EMOTIONS
#delete row 0.000
data_hist = data_hist.drop([0])

# show data (frequency)
print(data_hist)
data_hist.plot()
plt.show()


