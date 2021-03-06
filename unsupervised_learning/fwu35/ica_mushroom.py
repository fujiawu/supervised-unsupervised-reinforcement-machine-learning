from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, RandomizedPCA, FastICA
from random import shuffle
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
from load_mydata import LoadData
import math

mushroom = LoadData("mushroom")
data = scale(mushroom.data)
labels = np.array(mushroom.labels)

n_samples, n_features = data.shape
n_digits = len(np.unique(labels))
n_iter = 1000

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

t0 = time()
ica = FastICA(max_iter=n_iter)
reduced_data = ica.fit_transform(data)
print("time spent: %0.3fs" % (time()-t0))
#reduced_data = data

# Plot the data in ICA
fig=plt.figure()
#plt.clf()
n_plots=9
h = 0.02
x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
for index in range(1,n_plots+1):
   vert=math.floor(math.sqrt(n_plots))
   hori=n_plots/vert
   fig.add_subplot(vert,hori,index)
   i,j = 2*index-2, 2*index-1
   plt.plot(reduced_data[:, i], reduced_data[:, j], 'k.', markersize=2)
   #title = str(i) + ' vs ' + str(j)
   #plt.title(title)
   plt.xlabel('x'+str(i+1), fontsize=16)
   plt.ylabel('x'+str(j+1), fontsize=16)
   plt.xlim(x_min, x_max)
   plt.ylim(y_min, y_max)
   plt.xticks(())
   plt.yticks(())
plt.suptitle('ICA - mushroom.data', fontsize=18)
plt.show()
