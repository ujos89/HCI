#pca

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df1 = pd.read_pickle("./data/data_200922.pkl")
df2 = pd.read_pickle("./data/data_200922_2.pkl")
print(df1.shape, df2.shape)

df1 = df1[df1.Angry != 0]
df2 = df2[df2.Angry != 0]
print(df1.shape, df2.shape)

## data scaling (standard scaler)
# parameter add, index=list(df1.index.values) -> consider original index
scaler = StandardScaler()
df1_scaled = scaler.fit_transform(df1)
df1_scaled = pd.DataFrame(df1_scaled, columns = df1.columns)
df2_scaled = scaler.fit_transform(df2)
df2_scaled = pd.DataFrame(df2_scaled, columns = df2.columns)

print(df1_scaled)
print(df2_scaled)


##pca projection to 2D
pca = PCA(n_components = 2)
df1_pca = pca.fit_transform(df1_scaled)
df1_pca = pd.DataFrame(df1_pca, columns=['pc1','pc2'])
df2_pca = pca.fit_transform(df2_scaled)
df2_pca = pd.DataFrame(df2_pca, columns=['pc1','pc2'])

print(df1_pca)
print(df2_pca)

#draw graph
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.scatter(df1_pca['pc1'], df1_pca['pc2'], s=.1, alpha=.7)
ax2.scatter(df2_pca['pc1'], df2_pca['pc2'], s=.1, alpha=.7)

plt.show()

