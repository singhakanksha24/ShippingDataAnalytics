# Customer Analytics
Synopsis

Problem Statement – The Delivering of product on time is very important for company for maintain its customers because churning rate is also depends on  delivery time. Company also spent lots of money on delivery services. My aim is to Classify correctly whether product reached on time or not which helpful in making delivery services better and company will  spent money on delivery services according to delivery services.

Language : python
ML Algorithms : Decision Tree,Random Forest,AdaBoosting ,KNN,Gaussian Naïve Bayes, Logistic Regression, Multi-layer Perceptron.

Content with Explanation :
1.	Load the Dataset : Firstly , check the datatypes of data, if datatypes is miss-classified then use astype function and convert into correct datatype. Check for null values , then drop null values if null values less in percentage otherwise fill numeric values with mean and categorical values with mode.

2.	Check for outliers : Use boxplot or skew function to check  for the presence of outliers. Use Z-score or IQR method to remove the outliers.

3.	Visualization : Use various graphs and plots to visualize different variable relationship and get meaningful insights.

4.	Split the dataset : use train-test split to split the dataset into training and testing set. (split dataset into 3 sets to get rid from data leak)

5.	Use various Algorithms to make models :

I.	Decision Tree :
o	In a Decision tree, there are two nodes, which are the Decision Node and Leaf Node. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches. In order to build a tree, we use the CART algorithm, which stands for Classification and Regression Tree algorithm. A decision tree simply asks a question, and based on the answer (Yes/No), it further split the tree into subtrees.Find the best attribute in the dataset using Attribute Selection Measure (ASM).
      There are two popular techniques for ASM, which are:
o	Information Gain :  Information gain is the measurement of changes in entropy after the segmentation of a dataset based on an attribute.
o	Gini Index :  Gini index is a measure of impurity or purity used while creating a decision tree. An attribute with the low Gini index should be preferred as compared to the high Gini index.
II.	Random Forest : It is based on the concept of ensemble learning, which is a process of combining multiple Decision Tree Classifiers to solve a complex problem and to improve the performance of the model. Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset.
III.	AdaBoost Algorithm : short for Adaptive Boosting, is a Boosting technique used as an Ensemble Method. weights are re-assigned to each instance, with higher weights assigned to incorrectly classified instances. Boosting is used to reduce bias as well as variance for supervised learning. It works on the principle of learners growing sequentially. Except for the first, each subsequent learner is grown from previously grown learners. In simple words, weak learners are converted into strong ones. 

IV.	 KNN (K-Nearest Neighbor): K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories. It is also called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset.

Step-1: Select the number K of the neighbors
Step-2: Calculate the Euclidean distance of K number of neighbors
Step-3: Take the K nearest neighbors as per the calculated Euclidean distance.
Step-4: Among these k neighbors, count the number of the data points in each category.
Step-5: Assign the new data points to that category for which the number of the neighbor is maximum.
Step-6: Our model is ready.

V.	Gaussain Naïve Bayes :   Naive Bayes are a group of supervised machine learning classification algorithms based on the Bayes theorem. It is a simple classification technique, but has high functionality. They find use when the dimensionality of the inputs is high. One assumption taken is the strong independence assumptions between the features. These classifiers assume that the value of a particular feature is independent of the value of any other feature.

VI.	Multi-layer Perceptron :  It  is a class of feedforward artificial neural network (ANN). These models are called feedforward because information ﬂows through the function being evaluated from x, through the intermediate computations used to deﬁne  f, and ﬁnally to the output y. There are no feedback connections in which outputs of the model are fed back into itself. When feedforward neural networks are extended to include feedback connections, they are called recurrent neural networks. The MLPC employs backpropagation for learning the model. Technically, Spark used the logistic loss function for optimization and L-BFGS as an optimization routine. The number of nodes (say) N in the output layer corresponds to the number of classes.

VII.	Logistic Regression : Logistic regression is a process of modeling the probability of a discrete outcome given an input variable.  Logistic regression is a simple and more efficient method for binary and linear classification problems. It is a classification model, which is very easy to realize and achieves very good performance with linearly separable classes.

6.	Model selection According to importance metrics
7.	Show Importance features from dataset.





