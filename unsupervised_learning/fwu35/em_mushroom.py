from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from random import shuffle
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
from load_mydata import LoadData

def make_ellipses(gmm, ax):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipsns_e(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

mushroom = LoadData("mushroom")
data = scale(mushroom.data)
labels = mushroom.labels

n_samples, n_features = data.shape
n_digits = len(np.unique(labels))

print("n_digits: %d, \t n_samples: %d, \t n_features: %d"
      % (n_digits, n_samples, n_features))

print(79 * '_')
print('% 9s' %   'n_clusters  time   homo    compl  v-meas    ARI     AMI   silhouette')

def bench_em(estimator, data, labels):
    t0 = time()
    estimator.fit(data)
    estimator.labels_ = gmm.predict(data)
    print('    %i      %0.3fs  %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (len(estimator.means_), (time() - t0),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=300)))
#for i in range(2,40):
for i in range(2,3):
   gmm = GMM(n_components=i, covariance_type="full", n_iter=100)
   # spherical, diag, full, tied 
   bench_em(gmm, data, labels)

print(79 * '_')

#accuracy = np.mean(labels_ == labels)

##############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
gmm = GMM(n_components=15, covariance_type="full",
          n_iter=100)
gmm.fit(reduced_data)

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
Z = gmm.predict(np.c_[xx.ravel(), yy.ravel()])
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
centroids = gmm.means_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('EM clustering on Mushroom dataset after PCA\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
