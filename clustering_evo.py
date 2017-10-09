import csv
from sklearn.decomposition import PCA
from sklearn import metrics
import statistics
from sklearn.cluster import KMeans
import numpy as np
from sklearn import preprocessing
import matplotlib
matplotlib.use('QT4Agg')
from matplotlib import pyplot as plt

# The output of this will be a CSV file with each player tagged according to their scoring style
# as well as a visualization of the clustering algorithm

############################
# Import and organize data
############################
with open("nba_allszn_3ptera.csv", 'r') as myFile:
    dataLines = myFile.read().splitlines()

data_temp = []
for z in range(1, len(dataLines)):
    data_temp.append(dataLines[z].split(','))

data = []
for i in range(len(data_temp)):
    temp = []
    for j in range(2, 26):
        if data_temp[i][j] == '':
            temp.append(0)
        else:
            temp.append(float(data_temp[i][j]))
    temp.append((str(data_temp[i][0]) + ' ' + str(data_temp[i][1])))
    temp.append(str(data_temp[i][0]))

    data.append(temp)

print data[:3]

###############################################################################################
# scale and standardize data so that statistics on various scales don't have outsized impact
################################################################################################
train = data
temp = np.array(data)[:, 0:-3]
scaler = preprocessing.StandardScaler().fit(temp[:, 0:-3])
scorers = scaler.transform(temp[:, 0:-3]).tolist()
scorers = np.array(scorers)

print type(scorers)
print type(temp)

# look at variance between each of the principal components to determine how many dimensions to reduce into
pca = PCA(n_components=10)
pca.fit(scorers)
print(pca.explained_variance_ratio_)

# dimensionality reduction on data
reduced_data = PCA(n_components=2).fit_transform(scorers)
print ("reduced")


##########################
# Run KMeans clustering
##########################

# find optimal amount of kmeans clusters
def optimizer():
    scores = []
    for n in range(2, 20):
        ktest = KMeans(init='k-means++', n_clusters=n, n_init=10).fit(reduced_data)
        labels = ktest.labels_
        score = metrics.calinski_harabaz_score(reduced_data, labels)
        scores.append(score)

    print scores
    opt = scores.index(max(scores)) + 2
    return opt

optimizer_mode_set = []
for n in range(1):
    optimizer_mode_set.append(optimizer())
opt_mode = statistics.mode(optimizer_mode_set)
print opt_mode

# using optimal number of KMeans components, run KMeans clustering to get the "true" scorer styles
kmn = KMeans(init='k-means++', n_clusters=3, n_init=10)
clusters = kmn.fit_predict(reduced_data)
print clusters

# put clustering results into a dictionary with
# the keys being styles and values being players classified into those styles
classes = {}
for n in range(len(scorers)):
    if clusters[n] not in classes:
        classes[clusters[n]] = []
    classes[clusters[n]].append(train[n][-2])
print classes

# Get array with each player tied to their cluster classification
classifications = []
for n in range(len(scorers)):
    classifications.append([train[n][-2], clusters[n], train[n][-1]])

# write scorer tags to a csv (could and probably should do this in Pandas, borrowing this from an older script of mine)
with open('scorer_tags.csv', 'w') as writefile:
    datawrite = csv.writer(writefile)
    datawrite.writerow(['Player', 'Category'])
    for player in classifications:
        datawrite.writerow(player)

dates = ['1979-80', '1980-81', '1981-82', '1982-83', '1983-84', '1984-85', '1985-86', '1986-87', '1987-88', '1988-89',
         '1989-90', '1990-91', '1991-92', '1992-93', '1993-94', '1994-95', '1995-96', '1996-97', '1997-98', '1998-99',
         '1999-00', '2000-01', '2001-02', '2002-03', '2003-04', '2004-05', '2005-06', '2006-07', '2007-08', '2008-09',
         '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17']


def cluster_viz(year):
    # visualize clusters
    # Get the boundaries of the graph
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

    plt.figure(1)

    reduced = []
    clust = []
    namer = []
    for n in data:
        if n[-1] == year:
            ind = data.index(n)
            reduced.append([reduced_data.T[0][ind], reduced_data.T[1][ind]])
            clust.append(clusters[ind])
            namer.append(n[-2])

    reduc = np.array(reduced)
    clusts = np.array(clust)

    print np.shape(reduc), np.shape(reduced_data)
    print type(clusters), type(clusts), np.shape(clusters), np.shape(clusts)
    print clusts, clusters

    # Plot the nodes using 2D coordinates of the PCA reduced data, color-coded based on cluster
    plt.scatter(reduc.T[0], reduc.T[1], s=50, c=clusts, cmap=plt.cm.spectral)

    plt.title('The NBA in ' + year)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.savefig(year + ".png", dpi=100)
    plt.show()

for ndate in dates:
    cluster_viz(ndate)
