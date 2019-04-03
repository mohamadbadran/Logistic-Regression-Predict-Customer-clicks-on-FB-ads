# Libraries import
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# import the dataset
dataset = pd.read_csv('Facebook_Ads_2.csv')

# STEP #2: EXPLORE/VISUALIZE DATASET

# Let's count the number of the people who clicked and did not click on the advertisment
click_on_ad = dataset[dataset['Clicked']==1]
did_not_click = dataset[dataset['Clicked']==0]

# Counting some values
print('Total = ',len(dataset))
print('Number of customers who clicked on Ad = ',len(click_on_ad ))
print('Percentage Clicked = ', 1.* len(click_on_ad)/len(dataset) * 100, "%")
print('Did not click =', len(did_not_click))
print('Percentage who did not Click = ', 1.* len(did_not_click)/len(dataset) * 100, "%")

# plot the scatterplot of 'Time Spent on Site' versus 'Salary'
sns.scatterplot(dataset['Time Spent on Site'], dataset['Salary'], hue = dataset['Clicked'])

# plotting the boxplot to see the average salary of the people who clicked and who did not click on the ad
plt.figure(figsize = [5,5])
sns.boxplot(dataset['Clicked'], dataset['Salary'])

# plotting the boxplot to see the average 'Time Spent on Site' of the people who clicked and who did not click on the ad
plt.figure(figsize = [5,5])
sns.boxplot(dataset['Clicked'], dataset['Time Spent on Site'])

# Salary histogram
dataset['Salary'].hist(bins = 40)

# 'Time Spent on Site' histogram 
dataset['Time Spent on Site'].hist(bins = 20)

# STEP #3: PREPARE THE DATA FOR TRAINING/ DATA CLEANING

#Let's drop the emails, country and names (we can make use of the country later!)
dataset.drop( ['emails' , 'Country', 'Names'], axis = 1 , inplace = True)

#Let's drop the target coloumn before we do train test split
y = dataset['Clicked'].values
X = dataset.drop('Clicked', axis = 1).values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# STEP#4: MODEL TRAINING

# Splitting the data into 80% training and 20% testing set
from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 22 )

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# STEP#5: MODEL TESTING

# predicting training set results
y_predict_train = classifier.predict(X_train)
y_predict_train

# Making the Confusion Matrix of training set to see our model performance
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm , annot = True, fmt = "d")

# Displaying our model results (training set)
from sklearn.metrics import classification_report
print(classification_report(y_train, y_predict_train))

# predicting testing set results
y_pred_test = classifier.predict(X_test)
y_pred_test

# Making the Confusion Matrix of testing set to see our model performance
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm , annot = True ,fmt = "d")

# Displaying our model results (testing set)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_test))

# STEP #6: VISUALIZING TRAINING AND TESTING DATASETS

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train

# Create a meshgrid ranging from the minimum to maximum value for both features
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

# plot the boundary using the trained classifier
# Run the classifier to predict the outcome on all pixels with resolution of 0.01
# Colouring the pixels with 0 or 1
# If classified as 0 it will be magenta, and if it is classified as 1 it will be shown in blue 
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('magenta', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# plot all the actual training points

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('magenta', 'blue'))(i), label = j)
    
plt.title('Facebook Ad: Customer Click Prediction (Training set)')
plt.xlabel('Time Spent on Site')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Testing set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('magenta', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('magenta', 'blue'))(i), label = j)
plt.title('Facebook Ad: Customer Click Prediction (Testing set)')
plt.xlabel('Time Spent on Site')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

