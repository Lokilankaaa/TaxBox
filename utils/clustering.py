from scipy import cluster
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

data_path = 'res.txt'
r_index = []
feature = []
with open(data_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(',')
        r_index.append(line[0] + ' ' + line[1])
        feature.append([float(line[-2].split(':')[-1]), float(line[-1].split(':')[-1])])

feature = np.array(feature)
print(feature.shape)

res = SpectralClustering(n_clusters=2).fit(feature)
label = res.labels_

# centroid = cluster.vq.kmeans(cluster.vq.whiten(feature), 2)[0]
# label = cluster.vq.vq(feature, centroid)[0]
#
x0 = feature[label == 0, :]
x1 = feature[label == 1, :]
plt.scatter(x0[:, 0], x0[:, 1], c='red')
plt.scatter(x1[:, 0], x1[:, 1], c='blue')
# plt.show()
plt.savefig('cluster.png')
print(np.array(r_index)[(label == 0) if len(x0) < len(x1) else (label == 1)])
