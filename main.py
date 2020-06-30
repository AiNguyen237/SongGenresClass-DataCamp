#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from helper_functions import scaled_norm
#%%
# GET DATA 
# Loading the datasets into DataFrames 
tracks = pd.read_csv('/Users/ameliang/SongGenresClass-DataCamp/fma-rock-vs-hiphop.csv')
echo = pd.read_json('/Users/ameliang/SongGenresClass-DataCamp/echonest-metrics.json',precise_float=True)

# Merging the two dataframes into one using the 'track-id' columns and only keeping the 'genres_top' feature in tracks 
echonest_metrics = pd.merge(echo,tracks[['track_id','genre_top']],how='inner',on='track_id',left_index=True)
echonest_metrics.drop(['track_id'],axis=1, inplace=True)

# Splitting the data 
X = echonest_metrics[echonest_metrics.columns[:-1]]
y = echonest_metrics['genre_top']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=102)

# # Visually inspect the correlation between the features in the table echonest_metrics 
# # We want to use the features that don't have strong correlation to each other -> reducing feature redundancy
corr_metrics = echonest_metrics.corr()
plt.figure(figsize=(10,5))
sns.heatmap(corr_metrics)
plt.show()

#%%
# Normalize the data by transforming the data that has mean = 0 and standard deviation = 1 using normal distribution
X_train_norm =  (X_train-np.mean(X_train))/(np.std(X_train))
X_test_norm =  (X_test-np.mean(X_test))/(np.std(X_test))

#%%
# PCA for feature reductions
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_train_norm)
# Get explained variance ratios from PCA using all features 
exp_variances = pca.explained_variance_ratio_
# Plot the explained ratios using a barplot
fig, ax = plt.subplots()
ax.bar(range(pca.n_components_),exp_variances)
plt.xlabel('Principal Component')

#%%
# Calculate the cummulative explained variance 
cum_exp_variance = np.cumsum(exp_variances)
# Plot the cummulative and draw a dash line at 0.85
fig, ax = plt.subplots()
ax.plot(cum_exp_variance)
ax.axhline(y=0.85, linestyle = '--')

#%%
# Choose 5 components where about more than 85% of our variance explained 
n_components = 5
# Perform PCA with the chosen components and project data onto components
pca = PCA(n_components, random_state=10)
pca.fit(X_train_norm)
X_train_pca = pca.transform(X_train_norm)
X_test_pca = pca.transform(X_test_norm)

# %%
from sklearn import metrics
# Train a decision tree to classify the song genre
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=10)
tree.fit(X_train_pca,y_train)
tree_labels = tree.predict(X_test_pca)

print('Accuracy score for decision tree:', metrics.accuracy_score(y_test,tree_labels))
print('Classification reports:', metrics.classification_report(y_test,tree_labels))
print('Confusion matrix:', metrics.confusion_matrix(y_test,tree_labels))
# %%
# Compare a decision tree to a Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=10)
logreg.fit(X_train_pca,y_train)
logreg_labels = logreg.predict(X_test_pca)

print('Accuracy score for logistic regression:', metrics.accuracy_score(y_test,logreg_labels))
print('Classification reports:', metrics.classification_report(y_test,logreg_labels))
print('Confusion matrix:', metrics.confusion_matrix(y_test,logreg_labels))

# %%
# The data points for rock and hip hop are not distributed evenly
# Balancing datas to improve model bias

# Subsetting hip hop and rock only
hip_hop = echonest_metrics.loc[echonest_metrics['genre_top']=='Hip-Hop']
rock = echonest_metrics.loc[echonest_metrics['genre_top']=='Rock']

# Getting the same samples from each for balancing the data 
rock = rock.sample(len(hip_hop))

# Concat two dataframes together 
df_bal = pd.concat([hip_hop,rock])

# Defining the new train and test set 
features = df_bal.drop(['genre_top'],axis=1)
labels = df_bal['genre_top']
features_pca = pca.fit_transform(scaled_norm(features))

# %%
# Splitting the new features into train and test set
train_features, test_features, train_labels, test_labels = train_test_split(features_pca, labels, random_state=10)


# %%
# Create and train the new features and labels with Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
new_tree = DecisionTreeClassifier(random_state=10)
new_tree.fit(train_features,train_labels)
new_tree_labels = new_tree.predict(test_features)

print('Accuracy score for decision tree:', metrics.accuracy_score(test_labels,new_tree_labels))
print('Classification reports:', metrics.classification_report(test_labels,new_tree_labels))
print('Confusion matrix:', metrics.confusion_matrix(test_labels,new_tree_labels))

# %%
logreg = LogisticRegression(random_state=10)
logreg.fit(train_features,train_labels)
logreg_labels = logreg.predict(test_features)

print('Accuracy score for logistic regression:', metrics.accuracy_score(test_labels,logreg_labels))
print('Classification reports:', metrics.classification_report(test_labels,logreg_labels))
print('Confusion matrix:', metrics.confusion_matrix(test_labels,logreg_labels))


# %%
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(10)
tree_score = cross_val_score(new_tree,test_features,test_labels)
logreg_score = cross_val_score(logreg,test_features,test_labels)

# %%
print('Decision Tree:', np.mean(tree_score))

# %%
print('Logistic Regression:', np.mean(logreg_score))
