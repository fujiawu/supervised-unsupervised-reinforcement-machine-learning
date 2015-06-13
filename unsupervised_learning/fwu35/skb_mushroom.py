from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.decomposition import PCA, RandomizedPCA, FastICA
from random import shuffle
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
from load_mydata import LoadData
import math

mushroom = LoadData("mushroom")
#data = scale(mushroom.data)
data = np.array(mushroom.data)+1
labels = np.array(mushroom.labels)

n_samples, n_features = data.shape
n_digits = len(np.unique(labels))
n_iter = 1000

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))
t0 = time()
skb = SelectKBest(chi2, k=18)
reduced_data = skb.fit_transform(data, labels)
print("time spent: %0.3fs" % (time()-t0))
#reduced_data = data

# Plot the data
fig=plt.figure()
#plt.clf()
n_plots=9
h = 0.02
for index in range(1,n_plots+1):
   vert=math.floor(math.sqrt(n_plots))
   hori=n_plots/vert
   fig.add_subplot(vert,hori,index)
   i,j = 2*index-2, 2*index-1
   x_min, x_max = reduced_data[:, i].min()-1, reduced_data[:, i].max()+1
   y_min, y_max = reduced_data[:, j].min()-1, reduced_data[:, j].max()+1
   plt.plot(reduced_data[:, i], reduced_data[:, j], 'k.', markersize=7)
   #title = str(i) + ' vs ' + str(j)
   #plt.title(title)
   plt.xlabel('x'+str(i+1), fontsize=16)
   plt.ylabel('x'+str(j+1), fontsize=16)
   plt.xlim(x_min, x_max)
   plt.ylim(y_min, y_max)
   plt.xticks(())
   plt.yticks(())
plt.suptitle('Univariate Feature Selection - mushroom.data', fontsize=18)
plt.show()
