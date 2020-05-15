from sentence_array import sentence_array_final
from sklearn.cluster import SpectralClustering
#from sentence_array import doc_order
from num_clusters import cluster_range
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import collections
import matplotlib.cm as cm
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
#from yellowbrick.cluster import KElbowVisualizer

def document_generate():
    doc_array = sentence_array_final
    documents = []
    indices = []
    scores = []
    positions = []
    for doc in range(len(doc_array)):
        sentences = doc_array[doc][0]
        #print(sentences)
        score =  doc_array[doc][1]
        index = doc_array[doc][2]
        position = doc_array[doc][3]
        documents.append(sentences)
        positions.append([index,position,sentences])
        indices.append(index)
        scores.append(score)
    return documents,indices,positions,scores

def sort_cluster(clustering,scores):
    for i in range(len(clustering)):
        augmented_list = []
        for m in clustering[i]:
            #print(scores[m])
            augmented_list.append([m,scores[m]])
        augmented_list = sorted(augmented_list,key=lambda x: x[1],reverse=True)
        for n in range(len(augmented_list)):
            clustering[i][n] = augmented_list[n][0] 
    return clustering

def find_top_num(tfidf,scores,num):
    new_sum_matrix = tfidf.sum(axis=1)
    new_index = []
    final_index = []
    A = np.squeeze(np.asarray(new_sum_matrix))
    m = 0 
    for i in A:
        new_index.append([i,m])
        m += 1
    new_index = sorted(new_index,key=lambda x:x[0],reverse=False)
    indexes_of_centers = []
    i = 0
    number = 0
    #print("new_index",len(new_index))
    while(i < len(new_index) and number<=num):
        number += 1
        #print("selected",number-1,new_index[i][1])
        indexes_of_centers.append(new_index[i][1])
        i += math.floor(len(new_index)/(num-1))
        #print("new_i",i)
    if(number<num):
        indexes_of_centers.append(new_index[len(new_index)-1][1])
    for n in indexes_of_centers:
        #print(type(tfidf.getrow(n).toarray()))
        final_index.append(tfidf.getrow(n).toarray())
    #print("shape check",len(final_index[0]))
    return np.squeeze(np.array(final_index), axis=1)

def sentence_ordering(documents,clustering,indices,positions):
    summary = []
    index_array = []
    position_array = []
    result = False
    array_of_sentences = []
    # which_sentence_from_which_cluster = []
    # for i in range(len(clustering)):
    for i in range(len(clustering)):
        position_array.append(positions[clustering[i][0]])
        index_array.append(indices[clustering[i][0]])
        array_of_sentences.append(clustering[i][0])
    #print(str(index_array))
    if len(index_array) > 0 :
        result = index_array.count(index_array[0]) == len(index_array)
    #print(sorted(array_of_sentences))
    if(result):
        array_of_sentences = sorted(array_of_sentences)
        #print("array",array_of_sentences)
        for m in range(len(array_of_sentences)):
            summary.append(documents[array_of_sentences[m]])
    else:
        newlist = sorted(position_array, key=lambda x: x[1], reverse=False)
        #print(newlist[0][2])
        for ind in range(len(newlist)):
            summary.append(newlist[ind][2])
    #print(position_array)
    return summary
    #return 1 
def k_means():

    documents,indices,positions,scores = document_generate()
    #print('hulu',indices)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)
    #print("length",len(scores))
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(X)
    #print("debug",tfidf.shape)

    silhouette_avgs = []
    range_n_clusters = cluster_range

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 0.01])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, tfidf.shape[0] + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        #init_clusters = find_top_num(tfidf,scores,n_clusters)
        clusterer = KMeans(n_clusters=n_clusters,random_state=10)
        cluster_labels = clusterer.fit_predict(tfidf)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(tfidf.toarray(), cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        silhouette_avgs.append(silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(tfidf, cluster_labels)

    #     y_lower = 10
    #     for i in range(n_clusters):
    #         # Aggregate the silhouette scores for samples belonging to
    #         # cluster i, and sort them
    #         ith_cluster_silhouette_values = \
    #             sample_silhouette_values[cluster_labels == i]

    #         ith_cluster_silhouette_values.sort()

    #         size_cluster_i = ith_cluster_silhouette_values.shape[0]
    #         y_upper = y_lower + size_cluster_i

    #         color = cm.nipy_spectral(float(i) / n_clusters)
    #         ax1.fill_betweenx(np.arange(y_lower, y_upper),
    #                           0, ith_cluster_silhouette_values,
    #                           facecolor=color, edgecolor=color, alpha=0.7)

    #         # Label the silhouette plots with their cluster numbers at the middle
    #         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    #         # Compute the new y_lower for next plot
    #         y_lower = y_upper + 10  # 10 for the 0 samples

    #     ax1.set_title("The silhouette plot for the various clusters.")
    #     ax1.set_xlabel("The silhouette coefficient values")
    #     ax1.set_ylabel("Cluster label")

    #     # The vertical line for average silhouette score of all the values
    #     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    #     ax1.set_yticks([])  # Clear the yaxis labels / ticks
    #     ax1.set_xticks([-0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.1,0.12,0.14])

    #     # 2nd Plot showing the actual clusters formed
    #     colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    #     ax2.scatter([tfidf.tocsr()[:, 0].todense()], [tfidf.tocsr()[:, 1].todense()], marker='.', s=30, lw=0, alpha=0.7,
    #                 c=colors, edgecolor='k')

    #     # Labeling the clusters
    #     centers = clusterer.cluster_centers_
    #     # Draw white circles at cluster centers
    #     ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
    #                 c="white", alpha=1, s=200, edgecolor='k')

    #     for i, c in enumerate(centers):
    #         ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
    #                     s=50, edgecolor='k')

    #     ax2.set_title("The visualization of the clustered data.")
    #     ax2.set_xlabel("Feature space for the 1st feature")
    #     ax2.set_ylabel("Feature space for the 2nd feature")

    #     plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
    #                   "with n_clusters = %d" % n_clusters),
    #                  fontsize=14, fontweight='bold')

    # plt.show()

    min = -999
    index_of_clusters = -1
    for m in range(len(silhouette_avgs)):
        if(silhouette_avgs[m] > min):
            min = silhouette_avgs[m]
            index_of_clusters = m
    # K = range(4,10)
    # for k in K:
    #     kmeanModel = KMeans(n_clusters=k)
    #     kmeanModel.fit(tfidf)
    #     distortions.append(kmeanModel.inertia_)
    
    num_clusters = index_of_clusters + 3

    init_clusters = find_top_num(tfidf,scores,num_clusters)
    #print("shapedfgh",init_clusters.shape)
    #num_clusters = 5
    #print(num_clusters)
    km = KMeans(n_clusters=num_clusters,random_state = 42,init=init_clusters)
    km.fit(tfidf)
    # plt.figure(figsize=(16,8))
    # plt.plot(K, distortions, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Distortion')
    # plt.title('The Elbow Method showing the optimal k')
    # plt.show()
    clustering = collections.defaultdict(list)
    for idx, label in enumerate(km.labels_):
        clustering[label].append(idx)

    
    #summary = sentence_ordering(documents,clustering,indices,positions)
    summary = []
    for cl in range(len(clustering)):
        #print(clustering[cl][0])
        summary.append(documents[clustering[cl][0]])
    summary = sentence_ordering(documents,clustering,indices,positions)
    return summary


def main():
    file = open('summary.py','w')
    Summary = k_means()
    print("The summary is:")
    Summary = '. '.join(Summary)
    print(Summary)
    print("Word count:")
    print(len('.'.join(Summary).split(' ')))
    file.write("summary = " + repr(Summary))
    file.close()

if __name__=='__main__':
    main()

