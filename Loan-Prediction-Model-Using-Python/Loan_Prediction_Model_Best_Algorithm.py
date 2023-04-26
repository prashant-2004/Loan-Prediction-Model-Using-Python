import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB



# Import Train DataSet..
dataset = pd.read_csv("train_dataset.csv")

# Explore DataSet
print("HEAD OF Training Dataset : \n", dataset.head())

# Find the Shape of data in Rows and Column
print("\nSHAPE of Training Dataset : \n", dataset.shape)

# description of our Dataset..
print("\nDESCRIPTION of Training Dataset :\n",dataset.describe())

# We see how Credit_History Affects the Loan_Status,
# As Output says, If Credit_history = 1, then it has more chances to be eligible for loan.To check this, we use this.
print("\nCHECK is Credit_History will affect Loan_Status : \n",pd.crosstab(dataset['Credit_History'], dataset['Loan_Status'], margins=True))

# visualize by  plotting of ApplicantIncome from our database -
dataset.boxplot(column="ApplicantIncome")
# visualize by  Histogram of ApplicantIncome from our database
dataset["ApplicantIncome"].hist(bins=20)
dataset["CoapplicantIncome"].hist(bins=20)

# We see how Education is related to the Applicant_Income,
# It will not affect to much, but It will somewhat affect that graduate applicants has higher salaries as compared
# to non-graduate
dataset.boxplot(column="ApplicantIncome", by="Education")

# visualize for LoanAmount -
dataset.boxplot(column="LoanAmount")
dataset["LoanAmount"].hist(bins=20)
# Normalize the LoanAmount By Log-Function by Numpy
dataset["LoanAmount_log"] = np.log(dataset["LoanAmount"])
dataset["LoanAmount_log"].hist(bins=20)

# Finding total no. of Missing Values in all columns by using isnull()
print("\nCHECK is there is any Missing values in the training Dataset : \n",dataset.isnull().sum())

# Handling Missing Values by filling them-
# We use the mode() function(For Categorical Column) to fill the missing values and it will place the obtained
# mode value in place of missing
# Ex. Gender is divided into Male and Female in this 2 categories, so we use the mode()[0] At index 0.
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)  # in Male/Female Categories
dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)  # in YES/NO Categories
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)  # in 0/1/2/3+ Categories
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace=True)  # in YES/NO Categories
# LoanAmount is a Numeric Values so we use the mean() for the LoanAmount
dataset.LoanAmount = dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log = dataset.LoanAmount_log.fillna(dataset.LoanAmount_log.mean())
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True)  # in 120/360/.. Categories
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)  # in 0/1 Categories
print("\nAFTER FILLING MISSING VALUES, CHECK MISSING VALUES : \n", dataset.isnull().sum())

# Write STEPS ON WORD FILE FROM THIS BELOW
# Normalize TotalIncome of Applicant using log() function from NumPy by Adding Applicant & co-Applicant Income
dataset['TotalIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncome_log'] = np.log(dataset['TotalIncome'])
# Visualize TotalIncome_log by Histogram
dataset['TotalIncome_log'].hist(bins=20)  # Values are completely Normalized


# x store all the independant Variables
x = dataset.iloc[:, np.r_[1:5, 9:11, 13:15]].values

# y store the dependant variable, in our dataset we have Loan_Status as an dependant
y = dataset.iloc[:, 12].values

print("\nINDEPENDANT VARIABLES :\n ", x)
print("\nDEPENDANT VARIABLE : \n", y)


# spliting the Train And Test dataset
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)  # into 80-20% ratio.

print("\nINDEPENDANT TRAINED DATASET : \n",X_train)
print("\nDEPENDANT TRAINED DATASET : \n",Y_train)

# LabelEncoder() - used to convert text into Numeric Format..
# labelEncoder_X to change text into numeric data from X_train..
labelEncoder_X = LabelEncoder()

# Labeling Numeric Value to the X_train Splitted Data.
for i in range(0, 5):
    X_train[:, i] = labelEncoder_X.fit_transform(X_train[:, i])
X_train[:, 7] = labelEncoder_X.fit_transform(X_train[:, 7])

# labelEncoder_Y to change text into numeric data from Y_train..
labelEncoder_Y = LabelEncoder()
Y_train = labelEncoder_Y.fit_transform(Y_train)
print("\nDEPENDANT TRAINED DATASET with ALL TEXT VALUES CONVERTED INTO NUMERIC VALUES : \n",Y_train)


for i in range(0, 5):
    X_test[:, i] = labelEncoder_X.fit_transform(X_test[:, i])
X_test[:, 7] = labelEncoder_X.fit_transform(X_test[:, 7])
# These Indexes 0 : 5 & 7 are independant variable which are used to predict the dependant one.

Y_test = labelEncoder_Y.fit_transform(Y_test)

# Scaling  Indepedant Training & testing Dataset
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)



# Applying Algorithm of Naive bayes Classifier which give 83% ACCURACY AMONG tested  algorithm for given data are
# Random Forest(78%), KNN(80%),Decision Tree(70%),etc.
# Logistic Regression & SVM Algorithms both also gives same accuracy as Naive Bayes(83%)

NBClassifier = GaussianNB()
NBClassifier.fit(X_train,Y_train)
y_pred = NBClassifier.predict(X_test)
print("\nPREDICTED Y based on given X_train & Y_train :\n ",y_pred)
# Naive Bayes has more accuracy than the DT Classifier
print("\nThe Accurarcy of Naive Bayes is : ", metrics.accuracy_score(y_pred, Y_test),"\n")


# APPLYING NAIVE BAYES ON THE TEST DATA. Repeat Above All process on the test Data.

print("\n\nAPPLYING THESE STEPS ON THE UNKNOWN DATASET test_dataset.csv\n")
testData = pd.read_csv("test_dataset.csv")
print("\nCHECK is there is any Missing values in the unknown testing Dataset : \n",testData.isnull().sum())

testData["Gender"].fillna(testData["Gender"].mode()[0], inplace= True)
testData["Dependents"].fillna(testData["Dependents"].mode()[0], inplace= True)
testData["Self_Employed"].fillna(testData["Self_Employed"].mode()[0], inplace= True)
testData["Loan_Amount_Term"].fillna(testData["Loan_Amount_Term"].mode()[0], inplace= True)
testData["Credit_History"].fillna(testData["Credit_History"].mode()[0], inplace= True)

testData.LoanAmount = testData.LoanAmount.fillna(testData.LoanAmount.mean())
print("\nAFTER FILLING MISSING VALUES, CHECK MISSING VALUES IN UNKNOWN DATASET : \n",testData.isnull().sum())

testData["LoanAmount_log"] = np.log(testData["LoanAmount"])

testData["TotalIncome"] = testData["ApplicantIncome"] + testData["CoapplicantIncome"]
testData["TotalIncome_log"] = np.log(testData["TotalIncome"])


test = testData.iloc[:, np.r_[1:5, 9:11, 13:15]].values
for i in range(0, 5):
    test[:, i] = labelEncoder_X.fit_transform(test[:, i])

print("\nINDEPENDANT VARIABLES FROM UNKNOWN DATASET WITH NUMERIC CONVERTED : \n",test)  # converted into numeric

test = ss.fit_transform(test)  # scaling the TestData

pred = NBClassifier.predict(test)
print("\nFINAL APPLICANTS ELIGIBLE FOR LOAN FROM TEST DATA \n", pred)