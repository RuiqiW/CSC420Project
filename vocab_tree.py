import cv2
import numpy as np
import os
import sys
import pickle

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.externals import joblib


def train_vocab_tree(image_dir, num_clusters=10, perform_pca=False):
    image_paths = [image_dir + i for i in os.listdir(image_dir)]

    # Compute features in all images from database
    des_list = []
    for path in image_paths:
        image = cv2.imread(path)
        surf = cv2.xfeatures2d.SURF_create()
        kp, des = surf.detectAndCompute(image, None)
        for row in range(des.shape[0]):
            des[row] = des[row] / np.linalg.norm(des[row])
        des_list.append((path, des))

    data_size = len(des_list)

    descriptors = des_list[0][1]
    for image, des in des_list:
        descriptors = np.vstack((descriptors, des))

    # PCA
    if perform_pca:
        pca = PCA(n_components=40)
        pca.fit(descriptors)
        reduced_descriptors = pca.transform(descriptors)
        joblib.dump(pca, 'pca_model.joblib')
    else:
        reduced_descriptors = descriptors

    # Cluster the descriptors from the images in the database
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(reduced_descriptors)
    joblib.dump(kmeans, 'kmeans_model.joblib')

    # code_book = kmeans.cluster_centers_

    # Assign each descriptor in database and query image to the closest cluster
    image_features = np.zeros((data_size, num_clusters))
    for i in range(data_size):
        if perform_pca:
            pred = kmeans.predict(pca.transform(des_list[i][1]))
        else:
            pred = kmeans.predict(des_list[i][1])
        for k in range(len(pred)):
            image_features[i][pred[k]] += 1

    # Build an inverted file index
    inverted_file_index = [[] for _ in range(num_clusters)]
    for i in range(data_size):
        for k in range(num_clusters):
            if (image_features[i][k] > 2):
                inverted_file_index[k].append(i)


    # Compute a bag-of-words (BoW) vector for each retrieved image and query.
    # This vector just counts the number of occurrences of each word. It has as
    # many dimensions as there are visual words. Weight the vector with tf-idf.
    total_counts = np.count_nonzero((image_features > 0), axis=0)
    # add 1 to avoid Nan
    idf = np.log((1.0 * data_size + 1) / (1.0 * total_counts + 1))

    weighted_features = image_features * idf.reshape(1, -1)

    for row in range(weighted_features.shape[0]):
        weighted_features[row] = weighted_features[row] / (np.linalg.norm(weighted_features[row]) + 1e-7)

    return idf, weighted_features, inverted_file_index


if __name__ == '__main__':
    image_dir, num_clusters, perform_pca = sys.argv[1:]
    idf, weighted_features, inverted_file_index = train_vocab_tree(image_dir, num_clusters, perform_pca)

    model_dict = {}
    model_dict['idf'] = idf
    model_dict['weighted_features'] = weighted_features
    model_dict['inverted_file_index'] = inverted_file_index

    pickle.dump(model_dict, 'vocab_tree_model.pkl')

