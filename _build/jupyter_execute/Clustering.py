#!/usr/bin/env python
# coding: utf-8

# =============================================================
# # 1. Setup
# =============================================================

# ## Importing Relevant Libraries

# #### General

# In[1]:


import pandas as pd
import numpy as np


# #### For scaling the data

# In[2]:


from sklearn.preprocessing import StandardScaler, normalize


# #### For clustering

# In[3]:


from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score


# #### For principal component analysis

# In[4]:


from sklearn.decomposition import PCA


# #### For plotting

# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================
# # 2. Read in the data
# =============================================================

# ### Read in data from PVL-Delta model

# In[6]:


df_pvl_delta = pd.read_csv("./data/parameter_igt_pvl_delta.csv")


# #### Inspect the data

# In[7]:


df_pvl_delta.head()


# In[8]:


plt.scatter(df_pvl_delta["LR"], df_pvl_delta["Outcome_Sens"])


# #### Processing
# 'SubID' column can be dropped as it is not used for analysis.

# In[9]:


df_pvl_delta.drop(columns=['SubID'], inplace=True)


# In[10]:


df_pvl_delta.head()


# ### Read in data from the ORL model

# In[11]:


df_orl = pd.read_csv("./data/parameter_igt_orl.csv")


# #### Inspect the data

# In[12]:


df_orl.head()


# #### Processing
# 'subjID' column in input file is actually the 'group' column based on its contents. Thus, rename the column as so.

# In[13]:


df_orl.rename(columns={'subjID':'group'}, inplace=True)
df_orl.head()


# ### Read in data from the VPP model

# In[14]:


df_vpp = pd.read_csv("./data/parameters_igt_vpp.csv")


# #### Inspect the data

# In[15]:


df_vpp.head()


# #### Processing
# 'SubID' column is dropped as before as it is not used for analysis.

# In[16]:


df_vpp.drop(columns=['SubID'], inplace=True)
df_vpp.head()


# =============================================================
# # 3. Standardise the data
# =============================================================

# #### Choose which data source to use

# In the paper, Lili concludes that the VPP model was the best-fitting model in terms of short-term prediction performance for the IGT task, as measured by one-step-ahead leave-one-out information criterion (LOOIC) [1]. Therefore, we will begin by analysing this dataset.

# In[17]:


chosen_df = df_vpp


# * The *'group'* column will not be used as a feature for the clustering analysis, so we will exclude this from the processing for now.

# In[18]:


train_df = chosen_df.drop(columns=["group"])


# We can describe the data in this DataFrame and look at the ranges of the variables and their distributions

# In[19]:


train_df.describe()


# #### Visualising the distributions of the different variables

# In[99]:


sns.pairplot(train_df)


# From the pair plot above, we can clearly see that many of the variables have either positively or negatively skewed distributions. There also appears to be many outliers in each distribution and the different parameters don't share a common scale.

# To remedy this and also to ensure sound clustering analysis, we are going to **standardise** the dataset.

# #### Scaling and normalising the features

# In[21]:


# Standardising the data
standardised_train_array = StandardScaler().fit_transform(train_df)

# Normalizing the data
normalised_nd_standardised_train_array = normalize(standardised_train_array, axis=0)

# Converting the scaled array back to a DataFrame
scaled_train_df = pd.DataFrame(normalised_nd_standardised_train_array, columns=train_df.columns)


# In[22]:


scaled_train_df.describe()


# #### Add the group feature back in

# Since the data is now fully processed and ready for clustering, we can add the *'group'* column back in as it will be needed later on.

# In[23]:


scaled_full_df = pd.concat([chosen_df["group"], scaled_train_df], axis=1)


# In[24]:


scaled_full_df.head()


# =============================================================
# # 4. Consideration of clustering algorithms and optimal number of clusters
# =============================================================

# There are a number of clustering algorithms which we could implement. Below, we try the *KMeans* and *Spectral Clustering* algorithms and analyse how these perform on the data in question. 
# 
# To determine which algorithm is more appropriate to the dataset, we will measure their performance using the *Silhouette Coefficient*. The Silhouette Coefficient of a data point quantifies how similar a data point is to its own cluster compared to other clusters. We will use the mean Silhouette Coefficient across all data points to compare the performace of algorithms.
# 
# The **Silhouette Coefficient** for a particular data point is calculated by the below formula:
# 
# $$s = \frac{b - a}{max(a, b)}$$
# 
# Where:
# 
#  - **a** is the mean distance between a data point and all other points in the same cluster. (the mean intra-cluster distance)
#  - **b** is the mean distance between a data point and all other points in the *next nearest cluster*. (the mean nearest-cluster distance)
# 
# The Silhouette score can assume values between -1 and 1. Scores closer to -1 indicate incorrect clustering whereas those nearer to +1 indicate highly dense clustering. Scores around zero are indicative of overlapping clusters. Therefore, a good indicator of what clustering algorithm to choose would be the one whose mean Silhouette Coefficient is nearest to 1 [5].

# #### Choose the number of clusters to test these algorithms on

# In[25]:


num_clusters_to_iter = range(2, 11)


# #### Choose the random seed number for reproducibility

# In[26]:


random_seed_val = 127


# #### Define the functions for calculating the silhouette scores of the different models

# In[27]:


def get_silhouette_scores_list(scaled_train_df, model_function, num_clusters_to_itr, random_seed_val, affinity_value=None):
    
    # Initialise empty list for the Silhouette Scores for KMeans model
    s_scores = []

    # Running algorithm and calculating Silhouette Score
    for k in num_clusters_to_iter:

        # Building the clustering model
        if affinity_value:
            model = model_function(n_clusters = k, random_state = random_seed_val, affinity = affinity_value)
            
        else:
            model = model_function(n_clusters = k, random_state = random_seed_val)
        
        # Training the model and storing the predicted cluster labels
        labels = model.fit_predict(scaled_train_df)

        # Evaluating the performance and adding score to list
        s_scores.append(silhouette_score(scaled_train_df, labels))

    return s_scores


# In[28]:


def analyse_silhouette_scores_to_get_optimal_cluster_number(silhouette_score_list, num_clusters_to_iter):
    
    # Analyse these results to see the optimal cluster number
    max_silhouette_score = max(silhouette_score_list)
    index_max_score = silhouette_score_list.index(max_silhouette_score)

    print("The maximum Silhouette Score was", max_silhouette_score)
    print("This score is achieved by running", num_clusters_to_iter[index_max_score], "clusters")

    # plot the silhoutte scores against the cluster numbers
    plt.plot(num_clusters_to_iter, silhouette_score_list, 'bx-')
    plt.xlabel('# Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette scores for the model accross a different number of clusters')
    plt.show()

    # Output a table of the score for each cluster
    cluster_nd_score_df = pd.DataFrame(silhouette_score_list, index=num_clusters_to_iter).rename(columns={0: "Silhouette Score"})
    cluster_nd_score_df.index.name = "# Clusters"
    
    return cluster_nd_score_df.T


# ## Silhouette Scores for the KMeans Algorithm

# Below the KMeans algorithm is executed with various different numbers of clusters (from 2 clusters to 10). The mean Silhouette Coefficient (or more simply, the *Silhouette Score*) for the clusters produced is calculated for each execution of the algorithm. These scores are plotted below.

# #### Use K-Means to calculate the Silhouette Score for 2-10 clusters

# In[29]:


s_scores_km = get_silhouette_scores_list(scaled_train_df, KMeans, num_clusters_to_iter, random_seed_val)


# #### Analyse these scores and output & plot the score

# In[30]:


cluster_nd_score_km_df = analyse_silhouette_scores_to_get_optimal_cluster_number(s_scores_km, num_clusters_to_iter)
cluster_nd_score_km_df


# ## Silhouette Scores for the Spectral Clustering Algorithm

# In order to explore more options for clustering, we are going to test the performance of two different implementations of the Spectral Clustering algorithm. One implementation constructs the *affinity matrix* using a radial basis function (RBF) kernel (aka Gaussian kernel) while the other constructs this matrix by computing a graph of nearest neighbors.

# ### Spectral Clustering with *affinity='rbf'*

# #### Use Spectral Clustering to calculate the Silhouette Score for 2-10 clusters

# In[31]:


s_scores_rbf = get_silhouette_scores_list(scaled_train_df, SpectralClustering, num_clusters_to_iter, random_seed_val, 'rbf')


# #### Analyse these scores and output & plot the score

# In[32]:


cluster_nd_score_rbf_df = analyse_silhouette_scores_to_get_optimal_cluster_number(s_scores_rbf, num_clusters_to_iter)
cluster_nd_score_rbf_df


# ### Spectral Clustering with *affinity='nearest_neighbors'*

# #### Use Spectral Clustering to calculate the Silhouette Score for 2-10 clusters

# In[33]:


s_scores_nn = get_silhouette_scores_list(scaled_train_df, SpectralClustering, num_clusters_to_iter, random_seed_val, 'nearest_neighbors')


# #### Analyse these scores and output & plot the score

# In[34]:


cluster_nd_score_nn_df = analyse_silhouette_scores_to_get_optimal_cluster_number(s_scores_nn, num_clusters_to_iter)
cluster_nd_score_nn_df


# As we can see from the above results, running these clustering algorithms on this data using **two clusters** produces the best Silhouette Scores for all three algorithms.
# 
# One reason that we can see why this might be the case is that the sample of data subjects can be segmented into two groups as we saw at the start; *'young'* and *'old'* and these two clusters could just be encapsulating these two groups.
# To test this, we will attempt to visualise these clusters.

# From the above ouputs, we can see that all three algorithms get the same Silhouette Score for two clusters. As a result, we could choose any of the algorithms to test our hypothesis.
# 
# As the Spectral Clustering algorithm is more robust than KMeans, we will test this first, constructuing the affinity matrix using a radial basis function kernel (*affinity='rbf'*).

# #### Choose the number of clusters to use

# In[35]:


num_clusters = 2


# =============================================================
# # 5. Cluster the data using 2 clusters
# =============================================================

# #### Build the 'rbf' spectral clustering model with these 2 clusters

# In[36]:


spectral_model = SpectralClustering(n_clusters=num_clusters, affinity='rbf', random_state=random_seed_val)


# #### Training the model and storing predicted cluster labels

# In[37]:


cluster_labels_array = spectral_model.fit_predict(scaled_train_df)


# #### Turn this array to a dataframe so it can be concatenated with the rest of the data

# In[38]:


cluster_labels_df = pd.DataFrame(cluster_labels_array, columns=["Cluster"])


# In[39]:


cluster_labels_df.head()


# #### Add a new column to the original data with the cluster each data point is in

# In[40]:


full_df_nd_cluster = pd.concat([chosen_df, cluster_labels_df], axis=1)


# In[41]:


full_df_nd_cluster.head()


# Now that we have clustered this data based on the parameters, we want to verify our hypothesis that these 2 clusters will just contain the 'young' and 'old' groups exclusively.
# 
# We want to analyse these clusters to see how the 'old' and 'young' fall into these clusters and test this.

# =============================================================
# # 6. Analyse these 2 clusters
# =============================================================

# #### Create function to see how the young and old people are distributed within each cluster

# In[42]:


def see_how_the_young_and_old_fall_into_the_clusters(input_df):
    
    # copy the dataframe so that we arent overwriting the input dataframe
    df = input_df.copy()
    
    # Create a dataframe showing how many people fell into each cluster
    all_cluster_df = pd.DataFrame(df[["group", "Cluster"]].groupby("Cluster").count()).rename(columns={"group": "All"})
    
    # Change 'young' to 1 and 'old' to np.nan in the group column
    df["group"] = df["group"].replace('old', np.nan).replace('young', 1)
    
    # Create a dataframe showing how many young people fell into each cluster
    young_cluster_df = pd.DataFrame(df[["group", "Cluster"]].groupby("Cluster").count()).rename(columns={"group": "Young"})
    
    # Create a dataframe showing how many old people fell into each cluster
    old_cluster_df = pd.DataFrame(df["group"].isnull().groupby(df["Cluster"]).sum().astype(int)).rename(columns={"group": "Old"})
    
    # Output how many 'young' and 'old' people fell into each cluster
    return pd.merge(all_cluster_df, pd.merge(young_cluster_df, old_cluster_df, how="inner", on="Cluster"), how="inner", on="Cluster")


# #### Check the distribution of young and old people among the 2 clusters

# In[43]:


see_how_the_young_and_old_fall_into_the_clusters(full_df_nd_cluster)


# As you can see from the above table, our hypothesis was correct and the algorithm clustered the individuals into their respective age groups of 'young' and 'old'.
# 
# While this test did not provide us with any more information than we already had, we felt it was vital to not make any assumptions and verify the results we expected to see.
# So, even though from our analysis using the *Silhouette Score* we identified that two clusters maximised this score; this is because of how distinct the two groups of subjects are when it comes to decision-making and does not provide us with much insight. As a result, we are going to experiment with a different number of clusters and see if we can segement the subject group based on the different cognitive processes underlying the decision choices.

# =============================================================
# # 7. Re-consider the clustering algorithms and the number of clusters
# =============================================================

# So we have experimented using two clusters and have decided that this does not provide us with enough aditional information to satisfy our analysis.
# 

# #### Create function to output the silhoette scores excluding two clusters

# In[44]:


def take_in_a_dataframe_of_an_algorithms_silhouette_scores_and_output_analysis_excluding_cluster_2(cluster_nd_score_df):
    
    # remove cluster 2 from this dataframe
    cluster_nd_score_df_no_2 = cluster_nd_score_df.drop(columns=[2], axis=1)

    # get the silhouette scores and cluster number
    s_scores_without_2 = list(cluster_nd_score_df_no_2.iloc[0, :])
    num_clusters_to_iter_without_2 = cluster_nd_score_df_no_2.columns

    # analyse these scores
    return analyse_silhouette_scores_to_get_optimal_cluster_number(s_scores_without_2, num_clusters_to_iter_without_2)


# ## Silhouette Scores for the KMeans Algorithm (excluding two clusters)

# In[45]:


take_in_a_dataframe_of_an_algorithms_silhouette_scores_and_output_analysis_excluding_cluster_2(cluster_nd_score_km_df)


# ## Silhouette Scores for Spectral Clustering Algorithm (excluding two clusters)

# ### Spectral Clustering with *affinity='rbf'* (excluding two clusters)

# In[46]:


take_in_a_dataframe_of_an_algorithms_silhouette_scores_and_output_analysis_excluding_cluster_2(cluster_nd_score_rbf_df)


# ### Spectral Clustering with *affinity='nearest_neighbors'* (excluding two clusters)

# In[47]:


take_in_a_dataframe_of_an_algorithms_silhouette_scores_and_output_analysis_excluding_cluster_2(cluster_nd_score_nn_df)


# Based on these results, it is clear that the next maximum Silhouette Score comes from the KMeans algorithm as when we exclude two clusters from this analysis, using six clusters produces the next maximum silhouette score of nearly 0.27.
# 
# As a result of this, we propose to analyse the data using the KMeans algorithm with 6 clusters to see how the subjects fall into these clusters.
# 
# When we used 2 clusters last time, we didn't get much added information as the subjects ended up just being clustered into their respective age groups. Whereas when we use 6, we expect there to be sub-clusters within these different groups that will serve to detect more specific groups of people who have more dissimilar ways of making decisions.

# ## Further verification of optimal number of clusters using elbow curve of distortions

# #### Calculate distortion scores for different numbers of clusters

# In[48]:


distortions = []
num_clusters_to_iter = range(2,11)
for k in num_clusters_to_iter:
    kmeanModel = KMeans(n_clusters=k).fit(scaled_train_df)
    distortions.append(sum(np.min(cdist(scaled_train_df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / scaled_train_df.shape[0])


# #### Plot the elbow curve of the distortion that each cluster gives

# In[49]:


plt.plot(num_clusters_to_iter, distortions, 'bx-')
plt.xlabel('# Clusters')
plt.ylabel('Distortion')
plt.title('The Elbow Method displaying the optimal number of clusters')
plt.show()


# As we can see above, the rate of decrease in distortion begins to diminish once it passes 6 clusters. This concurs our choice of six as the optimal number of clusters.

# #### Re-choose the number of clusters

# In[50]:


num_cluster_second_time = 6


# =============================================================
# # 8. Re-cluster the data using 6 clusters
# =============================================================

# #### Building the KMeans model and clustering the data

# In[51]:


kmeans_model = KMeans(n_clusters=num_cluster_second_time, random_state=random_seed_val)


# #### Training the model and storing predicted cluster labels

# In[52]:


second_cluster_labels_array = kmeans_model.fit_predict(scaled_train_df)


# #### Get the centroids of these clusters

# In[53]:


centroids = kmeans_model.cluster_centers_


# In[54]:


centroids_df = pd.DataFrame(centroids, columns=scaled_train_df.columns)


# In[55]:


centroids_df


# #### Turn the labels array into a DataFrame so it can be concatenated with the rest of the data

# In[56]:


second_cluster_labels_df = pd.DataFrame(second_cluster_labels_array, columns=['Cluster'])


# In[57]:


second_cluster_labels_df.head()


# In order to visualise these clusters, we will need to reduce the dimensionality of the original Dataframe so that we can represent each data point in a two dimensional space. This can be achieved by performing **Principal Component Analysis** on the data. This is done below.

# =============================================================
# # 9. Principal Component Analysis
# =============================================================

# There are eight parameters in this model so principal component analysis is performed below to encapsulate this information into **three principal component axes**.

# #### Add the centroid data to the scaled data

# In[58]:


individuals_nd_centroids_df = pd.concat([scaled_train_df, centroids_df], axis=0)


# #### Inspect dataframe which PCA is being performed on

# In[59]:


individuals_nd_centroids_df.head()


# #### Use PCA to project the data to 3 dimensions

# In[60]:


# set the number of components
pca = PCA(n_components=3)

# create an array transforming the daa into these 3 components
principal_components_array = pca.fit_transform(individuals_nd_centroids_df)

# turn this array to a dataframe
principal_components_df = pd.DataFrame(data = principal_components_array, columns = ['PC1', 'PC2', 'PC3'])


# In[61]:


principal_components_df.head()


# #### Look at how much variance these 3 axes explain

# In[62]:


print(pca.explained_variance_ratio_)

print("\nWhen projecting the data onto the three principal componets, approximately {}% of the variance in the original data is retained".format(round(pca.explained_variance_ratio_.sum(), 3)*100))


# #### Remove the centroid data from this table

# In[63]:


individual_pca_df = principal_components_df.iloc[:-6, :]

pca_centroid_df = principal_components_df.iloc[-6:, :]


# #### Add these principal component columns and the cluster labels to the rest of the data

# In[64]:


final_full_scaled_df = pd.concat([scaled_full_df, individual_pca_df, second_cluster_labels_df], axis = 1)


# In[65]:


final_full_scaled_df.head()


# =============================================================
# # 10. Plot the 6 clusters using the PCA axes
# =============================================================

# ## Plot the clusters

# #### Choose the columns to plot

# In[66]:


pc1 = final_full_scaled_df['PC1']
pc2 = final_full_scaled_df['PC2']
pc3 = final_full_scaled_df['PC3']
group = final_full_scaled_df['group']
cluster = final_full_scaled_df['Cluster']
all_rows = pd.Series(["All"] * len(final_full_scaled_df))


# #### Check the minimum and maximum values across all axes

# In[67]:


print("pc1", min(pc1), max(pc1))
print("pc2", min(pc2), max(pc2))
print("pc3", min(pc3), max(pc3))


# #### Create function to visualise 3D projection

# In[92]:


def plot_3d_principal_component_points(targets, colours, column_to_colour_cluster_with, title, pca_centroid_df=pd.DataFrame()):
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # title the axes
    ax.set_xlabel('Principal Component 1', fontsize = 10)
    ax.set_ylabel('Principal Component 2', fontsize = 10)
    ax.set_zlabel('Principal Component 3', fontsize = 10)

    # set limits on the plot
    ax.set_xlim([-0.23, 0.4])
    ax.set_ylim([-0.2, 0.3])
    ax.set_zlim([-0.15, 0.3])
    ax.set_title(title)

    # plot the points
    for target, colour in zip(targets, colours):
        ax.scatter(pc1.loc[column_to_colour_cluster_with == target], pc2.loc[column_to_colour_cluster_with == target], pc3.loc[column_to_colour_cluster_with == target], c=colour, s=100, marker='o', edgecolors='grey')
    
    if not pca_centroid_df.empty:
        ax.scatter(pca_centroid_df['PC1'], pca_centroid_df['PC2'], pca_centroid_df['PC3'], c='black', s=100, marker='X')
        
    ax.legend(targets)
    plt.show()


# #### Plot the original data

# In[93]:


targets = ["All"]
colours = ['blue']

plot_3d_principal_component_points(targets, colours, all_rows, 'Original Data Projected Onto 3 Principal Component Axes')


# #### Plot the data colouring the different clusters

# In[94]:


targets = [0, 1, 2, 3, 4, 5]
colours = ['lime', 'red', 'blue', 'yellow', 'orange', 'c']

plot_3d_principal_component_points(targets, colours, cluster, 'Clusters Produced by KMeans Algorithm')


# #### Plot the data colouring the different clusters & plotting their centroids

# In[95]:


targets = [0, 1, 2, 3, 4, 5]
colours = ['lime', 'red', 'blue', 'yellow', 'orange', 'c']

plot_3d_principal_component_points(targets, colours, cluster, 'Clusters Produced by KMeans Algorithm with their Centroids', pca_centroid_df)


# #### Plot the data colouring the different groups

# In[98]:


targets = ['young', 'old']
colours = ['b', 'y']

plot_3d_principal_component_points(targets, colours,group, 'Original Data Segmented into Age Groups')


# =============================================================

# # 11. Analyse these 6 clusters

# =============================================================

# ## a.) See how the 'old' and 'young' fall into these clusters

# In[73]:


see_how_the_young_and_old_fall_into_the_clusters(final_full_scaled_df)


# ## b.) Characterising the clusters based on their different *cognitive attributes*
# As Lili states in the paper, a subject's performance on the IGT can be interpreted as a *synthesis of several different underlying psychological processes* [4]. As such, the parameters of the VPP model can be treated as neuro-psychologically interpretable variables. Some details on the semantics of each variable is given below:
# 
#  * **LR** $ \Leftrightarrow $ **Learning Rate**: Quantifies the rate at which the subject learns through experience.
#  * **Out_Sens** $ \Leftrightarrow $ **Outcome Sensitivity**: Accounts for the effects of win frequency. Subjects with a value less than 1 prefer decks with high win frequency over decks with the same long-term rewards but much less win frequency.
#  * **Res_Cons** $ \Leftrightarrow $ **Response Consistency**: Quantifies the consistency of the subjects choices.
#  * **Loss_Aver** $ \Leftrightarrow $ **Loss Aversion**: Responsible for the subjects sensitivity to loss relative to gains.
#  * **Gain_Impa** $ \Leftrightarrow $ **Gain Impact**: Quantifies the impact of gain on the perseverance behaviour of the subject.
#  * **Loss_Impa** $ \Leftrightarrow $ **Loss Impact**: Quantifies the impact of loss on the perseverance behaviour of the subject.
#  * **Deca_Rate** $ \Leftrightarrow $ **Decay Rate**: The decay parameter which controls how quickly decision makers forget their past deck choices.
#  * **RL_weight** $ \Leftrightarrow $ **Reinforcement Learning Weight**: Assumes values between 0 and 1. A low value indicates that the subject would rely less on reinforcement learning but more on the perseverance heuristic [3]. Whereas, a high value indicates the converse.
#  
# Based on these parameters, we can examine each cluster and characterise the decision-making tendencies of the subjects in that cluster. To do this, we need the original dataset before it was standardised; we append the cluster labels to this original DataFrame below.

# In[74]:


df_final = pd.concat([chosen_df, second_cluster_labels_df], axis=1)
df_final.head()


# #### Create functions to help analyse parameters

# The below functions will be useful for this piece of analysis. The first extracts the subjects from a specified cluster and provides a breakdown of the distribution of their parameters. While the second plots the distributions of each variable in a cluster.

# In[75]:


def analyse_cluster_params(df, cluster_label):
    cluster_df = df[df['Cluster'] == cluster_label].iloc[:, 1:-1]
    return cluster_df.describe()


# In[76]:


def plot_cluster_params(df, cluster_label, ax):
    cluster_df = df[df['Cluster'] == cluster_label].iloc[:, 1:-1]
    for i, col in enumerate(cluster_df.columns):
        sns.histplot(data=cluster_df, x=col, kde=True, ax=axes[i])
    plt.show()


# ### Summary statistics for entire subject group

# In[77]:


df_final.iloc[:, 1:-1].describe()


# ### Analsye Cluster 0

# In[78]:


analyse_cluster_params(df_final, 0)


# In[79]:


fig, axes = plt.subplots(ncols=8, figsize=(20,2))

plot_cluster_params(df_final, 0, axes)


# There are 34 subjects in this cluster, all of which are aged between 18 and 34 years (*group='young'*). The mean Outcome Sensitivity parameter of this group was 0.62; this indicates that subjects in this cluster prefered decks with high win frequency over decks with the same long-term rewards but much less win frequency. These subjects were less sensitive to win frequency than other clusters. 
# 
# The Gain and Loss Impact parameters also reveal some interesting characteristics of subjects in the cluster. With both distributions mainly encompassing negative values, this indicates that the feedback subjects received from deck choices, both good and bad, reinforced a tendency to switch from the chosen deck [3]. This coincides with the subjects tendency to *chase* high win frequency decks mentioned above.

# ### Analsye Cluster 1

# In[80]:


analyse_cluster_params(df_final, 1)


# In[81]:


fig, axes = plt.subplots(ncols=8, figsize=(20,2))

plot_cluster_params(df_final, 1, axes)


# The second cluster contains thirty one subjects; all aged between 65 and 88 years old (*group='old'*). We can likely expect some different characteristics between this cluster and the last due to the age difference of the subjects.
# 
# The standard deviation of the Learning Rate parameter of this cluster indicates that there is much more variability in the learning rates of the subjects in this cluster than in the last. The slightly higher values, on average, displayed in this cluster indicate that the most recent outcome has a large influence on the expectancy of the chosen deck and *forgetting is more rapid* [3]. 
# 
# The other notable differences in this cluster compared to the last is in the Gain Impact and Decay Rate parameters. Feedback, for subjects in this cluster, was far more likely to reinforce a tendency to persevere on the same deck on the next trial compared to subjects in the first cluster; as indicated by the much higher average Gain Impact value. The much lower average Decay Rate for subjects in this cluster however indicates rapid forgetting and a strong recency effect. This is probably to be expected based on age and Lili also makes this observation in the paper stating that the mean decay rate for subjects in older group is lower than that of the younger group [4].

# ### Analsye Cluster 2

# In[82]:


analyse_cluster_params(df_final, 2)


# In[83]:


fig, axes = plt.subplots(ncols=8, figsize=(20,2))

plot_cluster_params(df_final, 2, axes)


# The 38 subjects in this cluster, again, are all in the 'young' group. They exhibited very similar characteristics to the 'young' subjects in cluster 0 but there are a few things which set these clusters apart.
# 
#  * The most notable difference is in the Decay Rate parameter. Subjects in this cluster tended to have lower Decay Rates with a lower mean value, which could have been even lower due to the presence of one or two outliers. This indicates that subjects in this cluster forgot their past deck choices quicker than those is cluster 0, and exhibited a stronger recency effect.
#  * Interestingly, subjects in this group were more likely to persevere with the same deck choice when positive feedback was received, as indicated by the higher average Gain Impact parameter, compared to cluster 0. 
#  * The subjects were also less likely to switch decks upon receiving negative feedback (Loss Impact) compared to the 'young' subjects in cluster 0. The higher average value of Loss Impact is indicative of this.

# ### Analsye Cluster 3

# In[84]:


analyse_cluster_params(df_final, 3)


# In[85]:


#Plots are not needed here

df_final[df_final['Cluster'] == 3]


# There are only two subjects in this cluster, both from the 'old' group. It is unusual for a cluster this small to exist and it poses the question of what sets these two subjects apart from the other clusters?
# 
#  * It appears the Learning Rate of these two 'old' subjects is considerably higher than the average values in cluster 1  and cluster 5 of 0.098 and 0.094 respectively. So for these two subjects in particular, the most recent outcome had a large influence on the expectancy of the chosen deck and forgetting was more rapid [3]. These two values for learning rate are actually the highest values in the entire dataset, meaning these two subjects in particular forgot their previous choices more rapidly.
#  * The Response Consistency of 1.38 for the first subject was similar to the values present in the other two 'old' clusters. But the value of 1.79 for the second subject was actually the highest Response Consistency among all 'old' subjects. This indicates a high consistency in the choices of this particular subject.

# ### Analsye Cluster 4

# In[86]:


analyse_cluster_params(df_final, 4)


# 19 subjects make up the fifth cluster and interestingly, there is one 'old' subject in this cluster while the rest of the subjects are from the 'young' group. We will extract the subject from the 'old' group for reference.

# In[87]:


df_final[(df_final['Cluster'] == 4) & (df_final['group'] == 'old')]


# In[88]:


fig, axes = plt.subplots(ncols=8, figsize=(20,2))

plot_cluster_params(df_final, 4, axes)


# This cluster should display similar characteristics to the cluster 0 and cluster 2 as these both also contain exclusively subjects from the 'young' group (bar the one 'old' subject in this cluster, of course). The Learning Rate of subjects in this cluster was, on average, slightly lower than cluster 0 and cluster 2 (The actual mean of this parameter is skewed because of the outlier value present in the 'old' subject of 0.2). This indicates that subjects in this cluster placed less weight on past experiences of the chosen deck vs. the most recent selection from the deck. This also is indicative of the fact that subjects forgot past deck choices in a more gradual manner than in the other two 'young' groups. The most recent outcome had a smaller influence on the expectancy in the next trial, for these subjects.
# 
# The other most distinguishable characteristic of this cluster compared to the other 'young' groups was the much higher Loss Aversion tendencies of this cluster. The mean Loss Aversion parameter for subjects was 2.11 compared to 0.82 in cluster 0 and 0.85 in cluster 2. When Loss Aversion is greater than 1, the influence of a loss is greater than a gain on the subject [3]. This cluster of 'young' subjects, in particular, gave more weight to losses, compared to the other two 'young' clusters. 

# ### Analsye Cluster 5

# In[89]:


analyse_cluster_params(df_final, 5)


# In[90]:


fig, axes = plt.subplots(ncols=8, figsize=(20,2))

plot_cluster_params(df_final, 5, axes)


# There are 29 subjects in the final cluster, all from the 'old' group. They exhibit very similar characteristics to the subjects in cluster 1, which also exclusively contains 'old' subjects. This is to be expected. They do however, share some subtle differences with this other cluster.
# 
#  * The Loss Aversion of subjects in this cluster is slightly lower, on average. They were affected less by losses relative to gains than the other group, with a slightly lower mean value.
#  * The Gain and Loss Impacts respectively were both, on average, much lower for subjects in this cluster comaperd to those in cluster 1. These lower, negative values indicate that the feedback received for their deck choices reinforces a tendency to switch from the chosen deck.

# In[ ]:




