# skxplore
An exploratory top layer package for sklearn

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)  
This scikit-learn top layer package finds the machine learning model and hyperparameters best-suited for the dataset and properties the user has set. This is aimed to find the most appropriate machine learning model for the dataset and is not intended to substitute actual model engineering.

The find_model function has the following parameters:<br>
`dataset` takes a pandas dataframe as its value<br>
`train_size` is a float from the interval (0, 1); it defines the partition of the dataset for training and testing<br>
`problem` has three selections: classification, regression, and clustering<br>
`label` takes a column name in the pandas dataframe and treats it as the label for classification or the target value for regression<br>
`datatype` has two types: numerical and nominal (for text classification); numerical by default<br>
`dim_reduction` takes True or False as its value; it gives the option to apply dimensionality reduction on the dataset; False by default<br>
`features` is the number of components to remain after dimensionality reduction; auto by default<br>
`contains_negative` takes True or False as its value; setting its value to True uses principal component analysis, while setting its value to False uses non-negative matrix factorization; True by default<br>
`ensembling` takes True or False as its value; setting its value to True makes use of ensemble methods from base estimators, while setting its value to False disables ensembling<br>
`priority` has two selections: accuracy and time; selecting accuracy would enable the module to find for better hyperparameters and optimize the different algorithms in consideration; selecting time would use the default hyperparameters<br>

skxplore considers the following algorithms:<br>
1. Classification<br>
&nbsp;&nbsp;&nbsp;&nbsp;`Naive Bayes algorithm`, `K-nearest neighbors algorithm`, `Support vector machine classifier`, `eXtreme gradient boosting classifier`, and `Light gradient boosting machine classifier`
2. Regression<br>
&nbsp;&nbsp;&nbsp;&nbsp;`Lasso regression`, `Ridge regression`, `Elastic net regression`, `Linear regression`, `Support vector machine regressor`, `eXtreme gradient boosting regressor`, and `Light gradient boosting machine regressor`
3. Clustering<br>
&nbsp;&nbsp;&nbsp;&nbsp;`K-means clustering`, `Spectral clustering`, `Gaussian mixture model`, `Density-based spatial clustering of applications with noise (DBSCAN) algorithm`, and `Ordering points to identify the clustering structure (OPTICS) algorithm`

