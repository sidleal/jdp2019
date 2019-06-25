#! /usr/bin/env python3
import pandas as pd
from math import sqrt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import homogeneity_completeness_v_measure
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#keep to 3d
from mpl_toolkits.mplot3d import Axes3D


def calculate_wcss(data, max_grupos):
    wcss = []

    for n in range(2, max_grupos):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)

    return wcss


def optimal_number_of_clusters(wcss, max_grupos):
    print("----------------")
    print(wcss)
    x1, y1 = 2, wcss[0]
    x2, y2 = max_grupos -1, wcss[len(wcss)-1]

    print(x1,y1,x2,y2)

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)

    print(distances)
    print(distances.index(max(distances)))
    print("----------------")
    distances[0] = 0
    return distances.index(max(distances)) + 2


def run_experiments(dataset, dataset_name, num_grupos):

    max_grupos = num_grupos+1
    if len(dataset) < max_grupos:
        max_grupos = len(dataset)

    # calculando a soma dos quadrados para a quantidade de clusters
    sum_of_squares = calculate_wcss(dataset, max_grupos)

    # calculando a quantidade ótima de clusters
    n = optimal_number_of_clusters(sum_of_squares, max_grupos)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], dataset.iloc[:, 2])

    plt.title('RastrOS - %s: ' % dataset_name)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(2, max_grupos), sum_of_squares, 'b*-')
    ax.plot(n, sum_of_squares[n-2], marker='o', markersize=12,
        markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Número de Grupos')
    plt.ylabel('Média soma dos quadrados intra-grupo')
    plt.title('RastrOS - %s: Cotovelo KMeans' % dataset_name)
    plt.show()

    print("----->", n)

    kmeans = KMeans(n_clusters=num_grupos)
    kmeans.fit(dataset)

    #grafico 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    labels_kmeans = kmeans.labels_

    ax.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], dataset.iloc[:, 2],
               c=labels_kmeans.astype(np.float), edgecolor='k')

    result = {}
    for i in range(0, len(dataset)):
        ax.text(dataset.iloc[i-1, 0], dataset.iloc[i-1, 1], dataset.iloc[i-1, 2], i)
        result[i]=labels_kmeans[i-1]

    plt.title('RastrOS - %s: Textos - por índice' % dataset_name)
    plt.show()

    #grafico 3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], dataset.iloc[:, 2],
               c=labels_kmeans.astype(np.float), edgecolor='k')

    for i in range(0, len(dataset)):
        ax.text(dataset.iloc[i-1, 0], dataset.iloc[i-1, 1], dataset.iloc[i-1, 2], labels_kmeans[i-1])

    plt.title('RastrOS - %s: Grupos' % dataset_name)
    plt.show()

    print("kmeans - %s" % dataset_name, result)

    #----------------------- DBSCAN - Density -------------------------
    eps = 0.375
    if num_grupos > 15:
        eps = 0.32

    dbscan = DBSCAN(eps=eps, min_samples=1)
    dbscan.fit(dataset)
    labels_dbscan = dbscan.labels_
    result = {}
    for i in range(0, len(dataset)):
        result[i]=labels_dbscan[i-1]

    print("dbscan - %s - [%s]" % (dataset_name, eps), result)


    homo, comp, v_m = homogeneity_completeness_v_measure(labels_kmeans, labels_dbscan)
    print('Homogeneity (kmeans vs dbscan): {:.2f}%'.format(homo * 100))
    print('Completeness (kmeans vs dbscan): {:.2f}%'.format(comp * 100))
    print('V-Measure (kmeans vs dbscan): {:.2f}%'.format(v_m * 100))


    #----------------------- AgglomerativeClustering - Hierarchical -------------------------
    agglomer = AgglomerativeClustering(n_clusters=num_grupos)
    agglomer.fit(dataset)
    labels_agglomer = agglomer.labels_
    result = {}
    for i in range(0, len(dataset)):
        result[i]=labels_agglomer[i-1]

    print("agglomerative - %s " % (dataset_name), result)


    homo, comp, v_m = homogeneity_completeness_v_measure(labels_kmeans, labels_agglomer)
    print('Homogeneity (kmeans vs agglomer): {:.2f}%'.format(homo * 100))
    print('Completeness (kmeans vs agglomer): {:.2f}%'.format(comp * 100))
    print('V-Measure (kmeans vs agglomer): {:.2f}%'.format(v_m * 100))

    homo, comp, v_m = homogeneity_completeness_v_measure(labels_agglomer, labels_dbscan)
    print('Homogeneity (agglomer vs dbscan): {:.2f}%'.format(homo * 100))
    print('Completeness (agglomervs dbscan): {:.2f}%'.format(comp * 100))
    print('V-Measure (agglomer vs dbscan): {:.2f}%'.format(v_m * 100))



def main():
    df = pd.read_csv('rastros100_feats.csv')
    X = df.drop('index', axis=1)


    X_r = df[['sentences']]
    X_r.loc[:,'b'] = 0
    X_r.loc[:,'c'] = 0

    # print(X_r)

    # X_n = StandardScaler().fit_transform((X))
    # y = df.loc[:, 'index']

    # pca = PCA(n_components=3)
    # X_r = pca.fit(X).transform(X_n)


    #grafico 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2])
    ax.scatter(X_r.iloc[:, 0], X_r.iloc[:, 1], X_r.iloc[:, 1])

    plt.title('RastrOS - ')
    plt.show()

    X_jorn = X_r.iloc[0:71]
    # print(X_jorn, len(X_jorn))

    X_divu = X_r.iloc[72:]
    # print(X_divu, len(X_divu))

    # run_experiments(X_jorn, "Jornalístico", 35)
    # run_experiments(X_divu, "Divulgação Científica", 15)
    run_experiments(X_jorn, "Jornalístico", 6)
    run_experiments(X_divu, "Divulgação Científica", 8)



if __name__ == "__main__":
    main()
