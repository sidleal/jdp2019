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
    x1, y1 = 2, wcss[0]
    x2, y2 = max_grupos -1, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)

    return distances.index(max(distances)) + 2


def run_experiments(dataset, dataset_name, dataset_idx):

    max_grupos = 30
    if len(dataset) < max_grupos:
        max_grupos = len(dataset)

    # calculando a soma dos quadrados para a quantidade de clusters
    sum_of_squares = calculate_wcss(dataset, max_grupos)

    # calculando a quantidade ótima de clusters
    n = optimal_number_of_clusters(sum_of_squares, max_grupos)

    fig = plt.figure(figsize=[10,7])
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2])

    plt.title('RastrOS - %s ' % dataset_name)

    # fig = plt.figure(1)
    ax = fig.add_subplot(222)
    ax.plot(range(2, max_grupos), sum_of_squares, 'b*-')
    ax.plot(n, sum_of_squares[n-2], marker='o', markersize=12,
        markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Número de Grupos')
    plt.ylabel('Média soma dos quadrados intra-grupo')
    plt.title('Cotovelo KMeans')
    # plt.title('RastrOS - %s: Cotovelo KMeans' % dataset_name)
    # plt.show()

    print("--- %s - número ideal de grupos --> %s" %(dataset_name, n))

    kmeans = KMeans(n_clusters=n)
    kmeans.fit(dataset)

    #grafico 2
    # fig = plt.figure(1)
    ax = fig.add_subplot(223, projection='3d')
    labels_kmeans = kmeans.labels_

    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2],
               c=labels_kmeans.astype(np.float), edgecolor='k')

    result = {}
    for i in range(0, len(dataset)):
        ax.text(dataset[i-1, 0], dataset[i-1, 1], dataset[i-1, 2], dataset_idx[i])
        result[dataset_idx[i]]=labels_kmeans[i-1]

    plt.title('Textos - por índice')
    # plt.show()

    #grafico 3
    # fig = plt.figure(1)
    ax = fig.add_subplot(224, projection='3d')
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2],
               c=labels_kmeans.astype(np.float), edgecolor='k')

    for i in range(0, len(dataset)):
        ax.text(dataset[i-1, 0], dataset[i-1, 1], dataset[i-1, 2], labels_kmeans[i-1])

    plt.title('Grupos')
    plt.show()

    print("kmeans - %s" % dataset_name, result)

    #----------------------- DBSCAN - Density -------------------------
    eps = 1.1

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
    agglomer = AgglomerativeClustering(n_clusters=n)
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
    df = pd.read_csv('rastros100_feats.csv', index_col=0)
    # X = df.drop('index', axis=1)

    # complexidade estrutural do período (períodos simples vs. compostos)
    estruturais = ['words_per_sentence', 'sentences', 'words', 'sentence_length_max', 'sentence_length_min',
                   'sentence_length_standard_deviation', 'logic_operators', 'and_ratio', 'if_ratio', 'or_ratio',
                   'negation_ratio', 'conn_ratio', 'add_neg_conn_ratio', 'add_pos_conn_ratio', 'cau_neg_conn_ratio',
                   'cau_pos_conn_ratio', 'log_neg_conn_ratio', 'log_pos_conn_ratio', 'tmp_neg_conn_ratio',
                   'tmp_pos_conn_ratio', 'yngve', 'frazier', 'dep_distance', 'words_before_main_verb',
                   'clauses_per_sentence', 'prepositions_per_clause', 'adjunct_per_clause', 'prepositions_per_sentence',
                   'ratio_coordinate_conjunctions', 'ratio_subordinate_conjunctions', 'sentences_with_five_clauses',
                   'sentences_with_four_clauses', 'sentences_with_seven_more_clauses', 'sentences_with_six_clauses',
                   'sentences_with_three_clauses', 'punctuation_diversity', 'punctuation_ratio', 'non_svo_ratio',
                   'sentences_with_one_clause', 'sentences_with_two_clauses', 'sentences_with_zero_clause',
                   'adverbs_before_main_verb_ratio']

    # tipos de sentenças (ativas/passivas/relativas)
    tipos_sentenca = ['passive_ratio', 'relative_clauses', 'relative_pronouns_diversity_ratio', 'subordinate_clauses',
                      'infinite_subordinate_clauses', 'coordinate_conjunctions_per_clauses', 'apposition_per_clause']

    # mecanismos de construção de relações de correferência, entre outros
    correferencia = ['adjacent_refs', 'anaphoric_refs', 'adj_arg_ovl', 'arg_ovl', 'adj_stem_ovl', 'stem_ovl',
                     'adj_cw_ovl', 'adj_mean', 'adj_std', 'all_mean', 'all_std', 'givenness_mean', 'givenness_std',
                     'span_mean', 'span_std', 'coreference_pronoum_ratio']

    # informações sobre as categorias gramaticais: adjectives, adverbs, conjunctions, determiners, nouns, prepositions,
    # pronouns,  verbs; e flexão dos substantivos e verbos.
    morfossintaticas = ['adjective_ratio', 'adverbs', 'noun_ratio', 'verbs', 'pronoun_ratio', 'personal_pronouns',
                        'content_words', 'first_person_possessive_pronouns', 'first_person_pronouns', 'inflected_verbs',
                        'non-inflected_verbs', 'second_person_possessive_pronouns', 'second_person_pronouns',
                        'third_person_possessive_pronouns', 'third_person_pronouns', 'adjective_diversity_ratio',
                        'adjectives_max', 'adjectives_min', 'adjectives_standard_deviation', 'adverbs_diversity_ratio',
                        'adverbs_max', 'adverbs_min', 'adverbs_standard_deviation', 'verbal_time_moods_diversity',
                        'noun_diversity', 'nouns_max', 'nouns_min', 'nouns_standard_deviation', 'preposition_diversity',
                        'pronoun_diversity', 'pronouns_max', 'pronouns_min', 'pronouns_standard_deviation',
                        'indicative_condition_ratio', 'indicative_future_ratio', 'indicative_imperfect_ratio',
                        'indicative_pluperfect_ratio', 'indicative_present_ratio', 'indicative_preterite_perfect_ratio',
                        'subjunctive_future_ratio', 'indefinite_pronoun_ratio', 'oblique_pronouns_ratio',
                        'relative_pronouns_ratio', 'subjunctive_imperfect_ratio', 'subjunctive_present_ratio',
                        'abstract_nouns_ratio', 'verb_diversity', 'verbs_max', 'verbs_min', 'verbs_standard_deviation',
                        'syllables_per_content_word', 'demonstrative_pronoun_ratio']


    all_sel = estruturais + tipos_sentenca + correferencia + morfossintaticas

    print(len(estruturais), len(tipos_sentenca), len(correferencia), len(morfossintaticas), len(all_sel))


    X_r = df[all_sel]

    X_n = StandardScaler().fit_transform(X_r)

    pca = PCA(n_components=3)
    X_pca = pca.fit(X_r).transform(X_n)

    # grafico 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])
    plt.title('RastrOS - PCA ')
    plt.show()

    # gêneros
    jornalistico = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29,
                    39, 40, 41, 43, 47, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72]
    literario = [30, 31, 32, 33, 34, 35, 45, 51, 68]
    divulgacao = [7, 27, 36, 37, 38, 42, 44, 46, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 73, 74, 75, 76, 77, 78,
                  79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

    X_g = [
        ["Jornalístico", jornalistico],
        ["Literário", literario],
        ["Divulgação Científica", divulgacao]
    ]

    for genre in X_g:
        print(genre[0])

        subset = X_r.loc[genre[1]]

        X_n = StandardScaler().fit_transform(subset)
        #X_pca = pca.fit(X_r).transform(X_n)

        # X_pca = subset.values
        # X_pca = X_n

        run_experiments(X_n, genre[0], genre[1])



if __name__ == "__main__":
    main()
