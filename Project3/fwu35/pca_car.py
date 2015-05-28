from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, RandomizedPCA
from random import shuffle
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
from load_mydata import LoadData
import math

car = LoadData("car")
data = scale(car.data)
labels = np.array(car.labels)

n_samples, n_features = data.shape
n_digits = len(np.unique(labels))

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))
t0 = time()
pca = PCA()
#print(pca.get_params())
reduced_data = pca.fit_transform(data)
print("time spent: %0.3fs" % (time()-t0))
#print(pca.components_)
#print(pca.explained_variance_ratio_)
#print(pca.n_components_)
#print(pca.noise_variance_)

# Plot the data in PCA
fig=plt.figure()
#plt.clf()
n_plots=3
h = 0.02
x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
for index in range(1,n_plots+1):
   vert=math.floor(math.sqrt(n_plots))
   hori=n_plots/vert
   fig.add_subplot(vert,hori,index)
   i,j = 2*index-2, 2*index-1
   plt.plot(reduced_data[:, i], reduced_data[:, j], 'k.', markersize=4)
   #title = str(i) + ' vs ' + str(j)
   #plt.title(title)
   plt.xlabel('x'+str(i+1), fontsize=20)
   plt.ylabel('x'+str(j+1), fontsize=20)
   plt.xlim(x_min, x_max)
   plt.ylim(y_min, y_max)
   plt.xticks(())
   plt.yticks(())
plt.suptitle('PCA - car.data', fontsize=24)
plt.show()
