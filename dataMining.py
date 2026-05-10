import sys
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from operator import itemgetter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.naive_bayes import GaussianNB
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def wineClassification(trainFile,dataFile,outputFile):
	#create dataFrames for train and data files
	trainDf = pd.read_csv(trainFile,sep=';')
	dataDf =  pd.read_csv(dataFile,sep=';')

	#Separate train df to attributes 'Xtrain' and target 'ytrain'
	Xtrain = trainDf.drop("quality",axis=1)
	ytrain = trainDf["quality"]

	#Train model to data
	model = RandomForestClassifier()
	model.fit(Xtrain, ytrain)

	#predict and output results
	y_pred = model.predict(dataDf)
	dataDf = dataDf.assign(quality=y_pred)
	dataDf.to_csv(outputFile, sep=';', index=False)
	return

def clustering2D(dataFile, attr1, attr2, outputFile):
	#Create dataframe for the two attributes
	df =  pd.read_csv(dataFile,sep=';')
	user_select_attr1 = df[attr1]
	user_select_attr2 = df[attr2]
	clusterData1 = pd.concat([user_select_attr1,user_select_attr2],axis=1)

	#find optimal number of clusters using silhouette method
	range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
	silhouette_scores_array = []
	for n_clusters in range_n_clusters:
		clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
		cluster_labels = clusterer.fit_predict(clusterData1)
		silhouette_avg = silhouette_score(clusterData1, cluster_labels)
		silhouette_scores_array.append((n_clusters,silhouette_avg))
		#sample_silhouette_values = silhouette_samples(clusterData1, "void")
	n_clusters = max(silhouette_scores_array,key=itemgetter(1))[0]
	print("Ideal value of clusters is", n_clusters)

	#Perform Clustering
	kmeans = KMeans(n_clusters=n_clusters, n_init="auto", init='k-means++', random_state=42)
	y_kmeans = kmeans.fit_predict(clusterData1)
	
	#output results to dataframe and format cluster data to be used for plotting
	df['cluster'] = y_kmeans
	clusterData1 = clusterData1.to_numpy()

	#plot clusters
	colorsarray = ['red', 'green', 'blue', 'peru', 'purple', 'orange', 'magenta', 'deepskyblue', 'pink', 'black']
	for i in range(0,n_clusters):
		plt.scatter(clusterData1[y_kmeans==i,0], clusterData1[y_kmeans==i,1], s=3, c=colorsarray[i], label="Cluster "+str(i+1))

	#plot Centers
	plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, c='yellow', label='Centroids')
	# Graph settings
	plt.title('Clusters of wine quality data')
	plt.xlabel(attr1)
	plt.ylabel(attr2)
	plt.legend()
	plt.show()

	#output results to csv
	df.to_csv(outputFile, sep=';', index=False)
	return

def assocRuleMine(dataFile, outFile):
	df =  pd.read_csv(dataFile,sep=';')
	# Specify the continuous columns to discretize
	cols_to_discretize = ['alcohol', 'chlorides', 'citric acid', 'density', 'fixed acidity', 'free sulfur dioxide', 'pH', 'residual sugar', 'sulphates', 'total sulfur dioxide', 'volatile acidity']

	# Define number of bins
	num_bins = 5

	# Define bin labels
	bin_labels = ['bin{}'.format(i) for i in range(num_bins)]

	# Discretize the continuous columns and create separate columns for each bin
	for col in cols_to_discretize:
		bins = pd.cut(df[col], bins=num_bins, labels=bin_labels)
		binned_cols = pd.get_dummies(bins, prefix=col)
		df = pd.concat([df, binned_cols], axis=1)

	# Drop the original continuous columns
	df = df.drop(cols_to_discretize, axis=1)

	# Perform Apriori algorithm to generate frequent itemsets
	frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)

	# Generate association rules from frequent itemsets
	rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
	rules.to_csv(outputFile, sep=';', index=False)
	return

if __name__ == '__main__':
	argc = len(sys.argv)
	if argc < 2:
		print("Not enough arguments. Rerun with --help for more info.")
		exit(0)

	if sys.argv[1] == '--help' or sys.argv[1] == '-h':
		print("Usage: python3 dataMining.py [OPTION]...")
		print("Example: python3 dataMining.py -T train_data_file.csv input_data_file.csv output_data_file\n")
		print("OPTIONS:\n\t-T train_data_file.csv input_data_file.csv output_data_file\tPerforms Classification ")
		print("\t-C data_for_clustering.csv attribute1 attribute2 outputFile\tPerforms Clustering based on the 2 given attributes")
		print("\t-A data_for_assoc_rule.csv outputFile\t\t\t\tPerforms association rule mining and exports rules found to output file")
		exit(0)
	
	if sys.argv[1] == '-T':
		if argc != 5:
			print("Incorrect number of arguments for Classification. Use --help option for more info...")
			exit(0)
		trainFile = open(sys.argv[2],'r')
		dataFile = open(sys.argv[3],'r')
		outputFile = open(sys.argv[4],'w', newline='\n')
		wineClassification(trainFile,dataFile,outputFile)
		trainFile.close()
		dataFile.close()
		outputFile.close()
	elif sys.argv[1] == '-C':
		if argc != 6:
			print("Incorect number of arguments for Clustering. Use --help option for more info...")
			exit(0)
		dataFile = open(sys.argv[2],'r')
		outputFile = open(sys.argv[5],'w',newline='\n')
		
		clustering2D(dataFile, sys.argv[3], sys.argv[4], outputFile)
		dataFile.close()
		outputFile.close()
	elif sys.argv[1] == '-A':
		if argc != 4:
			print("Incorect number of arguments for Association Rule Learning. Use --help option for more info...")
			exit(0)
		dataFile = open(sys.argv[2],'r')
		outputFile = open(sys.argv[3],'w', newline='\n')
		assocRuleMine(dataFile, outputFile)
		dataFile.close()
		outputFile.close()

