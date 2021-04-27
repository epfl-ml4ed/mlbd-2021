import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from scipy.sparse.csgraph import laplacian
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph

SEED = 111


def spectral_clustering(X, n_clusters, affinity='precomputed'):
    """
    Spectral clustering
    :param X: np array of data points or affinity matrix
    :param n_clusters: number of clusters
    :param affinity: str indicating the type of input
    :return: tuple (kmeans, proj_X, eigenvals_sorted)
        WHERE
        kmeans scikit learn clustering object
        proj_X is np array of transformed data points
        eigenvals_sorted is np array with ordered eigenvalues

    """
    if affinity == 'nearest_neighbors':
        # Construct a similarity graph
        n_neighbors = 8
        connectivity = kneighbors_graph(X, n_neighbors=n_neighbors, mode='connectivity')
        adjacency_matrix = (1 / 2) * (connectivity + connectivity.T)

        # Compute the unnormalized graph Laplacian
        L = laplacian(csgraph=adjacency_matrix, normed=False)
        L = L.toarray()
    elif affinity == 'precomputed':
        # Assume affinity matrix has already been computed
        L = laplacian(X, normed=True)

        # Compute the first 𝑘 eigenvectors
    eigenvals, eigenvcts = linalg.eig(L)
    eigenvals = np.real(eigenvals)
    eigenvcts = np.real(eigenvcts)

    eigenvals_sorted_indices = np.argsort(eigenvals)
    eigenvals_sorted = eigenvals[eigenvals_sorted_indices]
    indices = eigenvals_sorted_indices[: n_clusters]
    proj_X = eigenvcts[:, indices.squeeze()]

    # Cluster the points using k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    kmeans.fit(proj_X)

    return kmeans, proj_X, eigenvals_sorted


def compute_bic(kmeans, X):
    """
    Computes the BIC metric

    :param kmeans: clustering object from scikit learn
    :param X: np array of data points
    :return: BIC
    """
    # Adapted from: https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels = kmeans.labels_
    # number of clusters
    k = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    # size of data set
    N, D = X.shape

    # compute variance for all clusters beforehand
    cl_var = (1.0 / (N - k) / D) * sum([sum(distance.cdist(X[np.where(labels == i)], \
                                                           [centers[0][i]], 'euclidean') ** 2) for i in range(k)])

    LL = np.sum([n[i] * np.log(n[i]) -
                 n[i] * np.log(N) -
                 ((n[i] * D) / 2) * np.log(2 * np.pi * cl_var) -
                 ((D / 2) * (n[i] - 1)) for i in range(k)])

    d = (k - 1) + 1 + k * D
    const_term = (d / 2) * np.log(N)

    BIC = LL - const_term

    return BIC


def plot_metrics(n_clusters_list, metric_dictionary):
    """
    Plots metric dictionary (auxilary function)
    [Optional]

    :param n_clusters_list: List of number of clusters to explore
    :param metric_dictionary:
    """
    fig = plt.figure(figsize=(12, 10), dpi=80)
    i = 1

    for metric in metric_dictionary.keys():
        plt.subplot(2, 2, i)

        if metric == 'Eigengap':
            clusters = len(n_clusters_list)
            eigenvals_sorted = metric_dictionary[metric]
            plt.scatter(range(1, len(eigenvals_sorted[:clusters * 2]) + 1), eigenvals_sorted[:clusters * 2])
            plt.xlabel('Eigenvalues')
            plt.xticks(range(1, len(eigenvals_sorted[:clusters * 2]) + 1))
        else:
            plt.plot(n_clusters_list, metric_dictionary[metric], '-o')
            plt.xlabel('Number of clusters')
            plt.xticks(n_clusters_list)
        plt.ylabel(metric)
        i += 1


def get_heuristics_spectral(X, n_clusters_list=range(2, 10)):
    """
    Calculates heuristics for optimal number of clusters with Spectral Clustering

    :param X: np array of data points
    :param n_clusters_list: List of number of clusters to explore
    """
    silhouette_list = []
    distortion_list = []
    bic_list = []
    eigengap_list = []

    for n in n_clusters_list:
        kmeans, proj_X, eigenvals_sorted = spectral_clustering(X, n, affinity='precomputed')
        y_pred = kmeans.labels_

        if n == 1:
            silhouette = np.nan
        else:
            silhouette = silhouette_score(proj_X, y_pred)
        silhouette_list.append(silhouette)

        distortion = kmeans.inertia_
        distortion_list.append(distortion)

        bic = compute_bic(kmeans, proj_X)
        bic_list.append(bic)

    metric_dictionary = {'BIC': bic_list,
                         'Distortion': distortion_list,
                         'Silhouette': silhouette_list,
                         'Eigengap': eigenvals_sorted}

    plot_metrics(n_clusters_list, metric_dictionary)
