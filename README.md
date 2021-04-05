Objectives:
(1) Load the dataset and perform splitting into training and validation sets with a 70:30  ratio. 

(2) Implement the K Means algorithm using sklearn. You need to find the optimal  number of clusters using the elbow method. Plot the error vs number of clusters graph  while using the elbow method. Report the optimal number of clusters found. 

(3) Use Scatter plot to visualize the dataset to depict the clusters formed (optimal). 

(4) Report the training and the validation accuracy. Comment on the accuracy  obtained for both the sets.


Solution:
 
Dataset used: We will use Iris dataset to perform this k-means algorithm.
Although we can use the dataset from (UCI Machine Learning Repository: Iris Data Set)
but here we loaded it directly using .load_iris of sklearn datasets

The number of rows in the data set is 150, and the number of columns are 5. First 4 columns contain features and the last column contain the label associated with them.
#data Bunch
Dictionary-like object, with the following attributes.

data{ndarray, dataframe} of shape (150, 4)
The data matrix. If as_frame=True, data will be a pandas DataFrame.

target: {ndarray, Series} of shape (150,)
The classification target. If as_frame=True, target will be a pandas Series.

feature_names: list
The names of the dataset columns.

target_names: list
The names of target classes.

frame: DataFrame of shape (150, 5)
Only present when as_frame=True. DataFrame with data and target.


#Elbow Method for determining number of clusters:

Here, currently we don't know the value of what K is to be taken, so we started trying with K=5, which for sure will divide the dataset into clusters, but what we want is to use the Elbow method, to get the optimal value of K.

Optimal number of clusters can be get as follows:

Compute clustering algorithm (e.g., k-means clustering) for different values of k. For instance, by varying k from 1 to 10 clusters.
For each k, calculate the total within-cluster sum of square (wss). We have to minimse this value.
Plot the curve of wss according to the number of clusters k.
The location of a bend (knee) in the plot is generally considered as an indicator of the appropriate number of clusters.

Splitting dataset into training set and validation set as well as training:
For this we will simply use train_test_split of sklearn.

Finding the number of clusters as 3 from elbow method, we need to split our dataset into a 70:30 ratio for training and validation purposes.

After that, we will simply use scikit learn to perform K-means algorithm in training dataset as well as validation dataset.

Scatter plot for actual dataset vs. dataset after clustering:

'''iris_df = pd.DataFrame(iris.data)
iris_df.columns = ['sepal_length','sepal_width','petal_length','petal_width']
y1 = pd.DataFrame(iris.target)
y1.columns =['Targets']
color_theme = np.array(['red','green','blue'])

plt.scatter(x=iris_df.petal_length, y=iris_df.petal_width, c= color_theme[iris.target],s=50)
plt.title ("This is Actual Flower Cluster")'''

Accuracy:
 We will print it for both traiing set andvalidation set.
 



