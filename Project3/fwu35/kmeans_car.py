from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from random import shuffle
from load_mydata import LoadData

car = LoadData("car")
data = scale(car.data)
labels = np.array(car.labels)

n_samples, n_features = data.shape
n_digits = len(np.unique(labels))

sample_size = 300

print("n_digits: %d, \t n_samples: %d, \t n_features: %d"
      % (n_digits, n_samples, n_features))

print(79 * '_')
print('% 9s' %   'n_clusters  time   homo    compl  v-meas    ARI     AMI   silhouette')

def bench_k_means(estimator, data, labels):
    t0 = time()
    estimator.fit(data)
    #print(estimator.labels_)
    #print(labels)
    print('    %i      %0.3fs  %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (len(estimator.cluster_centers_), (time() - t0), 
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=300)))

#for i in range(2,200):
for i in range(2,3):
  kmeans = KMeans(init="random", n_clusters=i, n_init=10)
  bench_k_means(kmeans, data, labels)

print(79 * '_')

###############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init="random", n_clusters=15, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#print(x_min, x_max, y_min, y_max)
#print(xx,yy)
#print(xx.ravel())
# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
#print(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=4)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on Car dataset after PCA\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
