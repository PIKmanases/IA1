import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

medicina = np.array([
    [8, 2], [9, 7], [2, 12], [9, 1], [10, 7], [3, 14], [8, 1], [1, 13]
])

r_kmeans = KMeans(n_clusters=3)
r_kmeans.fit(medicina)

print("Centroides")
print(r_kmeans.cluster_centers_)
print("No. Itereaciones")
print(r_kmeans.n_iter_)
print("Clasificación de los grupos")
print(r_kmeans.predict(medicina))

plt.scatter(medicina[:, 0], medicina[:, 1], c=r_kmeans.labels_, cmap='prism')
#plt.scatter(r_kmeans.cluster_centers_[:, 0], r_kmeans.cluster_centers_[:, 1], marker='+')
plt.show()