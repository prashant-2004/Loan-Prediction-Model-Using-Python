import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Import Train DataSet..
dataset = pd.read_csv("train_dataset.csv")

# Explore DataSet
print(dataset.head())

# Find the Shape of data in Rows and Column
print(dataset.shape)

# info and description of our Dataset..
print(dataset.info())
print(dataset.describe())

# We see how Credit_History Affects the Loan_Status,
# As Output says, If Credit_history = 1, then it has more chances to be eligible for loan.To check this, we use this.
print(pd.crosstab(dataset['Credit_History'], dataset['Loan_Status'], margins=True))

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
print(dataset.isnull().sum())

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
print(dataset.isnull().sum())

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

print(x)
print(y)


# spliting the Train And Test dataset
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)  # into 80-20% ratio.

print(X_train)

# LabelEncoder() - used to convert text into Numeric Format..
# labelEncoder_X to change text into numeric data from X_train..
labelEncoder_X = LabelEncoder()

# Labeling Numeric Value to the X_train Splitted Data.
for i in range(0, 5):
    X_train[:, i] = labelEncoder_X.fit_transform(X_train[:, i])
X_train[:, 7] = labelEncoder_X.fit_transform(X_train[:, 7])
print(X_train)


# labelEncoder_Y to change text into numeric data from Y_train..
labelEncoder_Y = LabelEncoder()
Y_train = labelEncoder_Y.fit_transform(Y_train)
print(Y_train)


for i in range(0, 5):
    X_test[:, i] = labelEncoder_X.fit_transform(X_test[:, i])
X_test[:, 7] = labelEncoder_X.fit_transform(X_test[:, 7])
# These Indexes 0 : 5 & 7 are independant variable which are used to predict the dependant one.
print(X_test)  # Display all Numeric Values

Y_test = labelEncoder_Y.fit_transform(Y_test)
print(Y_test)  # Display all Numeric Values


# Scaling  Indepedant Training & testing Dataset
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)


# Applying 1st Algorithm of Decision Tree Classifier
DTClassifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
DTClassifier.fit(X_train, Y_train)   # Providing algorithm the Training dataset
# Prediction by giving X_test Data to get correct y output..using DT Classifier Algorithm.
y_pred = DTClassifier.predict(X_test)
print(y_pred)
# DT Classifier Algorithm gives the 70% accuracy
print("\nThe Accurarcy of Decision Tree is : ", metrics.accuracy_score(y_pred, Y_test))



# Applying 2nd Algorithm of Naive bayes Classifier
NBClassifier = GaussianNB()
NBClassifier.fit(X_train,Y_train)
y_pred = NBClassifier.predict(X_test)
print(y_pred)
# Naive Bayes has more accuracy than the DT Classifier 83%
print("\nThe Accurarcy of Naive Bayes is : ", metrics.accuracy_score(y_pred, Y_test))


# Applying 3rd Algorithm of Random Forest Classifier
RFClassifier = RandomForestClassifier(n_estimators=100)
RFClassifier.fit(X_train,Y_train)
y_pred = RFClassifier.predict(X_test)
print(y_pred)
# Naive Bayes has more accuracy than the Random forest Regression
print("\nThe Accurarcy of Random forest Classfier is : ", metrics.accuracy_score(y_pred, Y_test))



# Applying 4th Algorithm of K-neighbors Classifier
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)
print("\nPREDICTED Y based on given X_train & Y_train :\n ",y_pred)
# Naive Bayes has more accuracy than the This KNN Algorithm -- KNN gives 80% accuracy
print("\nThe Accurarcy of KNN is : ", metrics.accuracy_score(y_pred, Y_test),"\n")


# Applying 5th Algorithm of Logistic regression Classifier
lg = LogisticRegression(random_state=0 , multi_class='auto', solver='lbfgs', max_iter=1000)
lg.fit(X_train,Y_train)
y_pred = lg.predict(X_test)
print("\nPREDICTED Y based on given X_train & Y_train :\n ",y_pred)
# Naive Bayes gives same accuracy with KNN Algorithm -- KNN gives 80% accuracy
print("\nThe Accurarcy of LOGISTIC REGRESSION is : ", metrics.accuracy_score(y_pred, Y_test),"\n")


# Applying 6th Algorithm of SVM Classifier
machine = SVC(random_state=0)
machine.fit(X_train,Y_train)
y_pred = machine.predict(X_test)
print("\nPREDICTED Y based on given X_train & Y_train :\n ",y_pred)
# SVM also gives same accuracy with Naive Bayes Algorithm -- KNN gives 80% accuracy
print("\nThe Accurarcy of SVM is : ", metrics.accuracy_score(y_pred, Y_test),"\n")




# APPLYING NAIVE BAYES ON THE TEST DATA. Repeat Above All process on the test Data.

testData = pd.read_csv("test_dataset.csv")
print(testData.isnull().sum())

testData["Gender"].fillna(testData["Gender"].mode()[0], inplace= True)
testData["Dependents"].fillna(testData["Dependents"].mode()[0], inplace= True)
testData["Self_Employed"].fillna(testData["Self_Employed"].mode()[0], inplace= True)
testData["Loan_Amount_Term"].fillna(testData["Loan_Amount_Term"].mode()[0], inplace= True)
testData["Credit_History"].fillna(testData["Credit_History"].mode()[0], inplace= True)

testData.LoanAmount = testData.LoanAmount.fillna(testData.LoanAmount.mean())
print(testData.isnull().sum())

testData["LoanAmount_log"] = np.log(testData["LoanAmount"])

testData["TotalIncome"] = testData["ApplicantIncome"] + testData["CoapplicantIncome"]
testData["TotalIncome_log"] = np.log(testData["TotalIncome"])

print(testData.head())

test = testData.iloc[:, np.r_[1:5, 9:11, 13:15]].values
for i in range(0, 5):
    test[:, i] = labelEncoder_X.fit_transform(test[:, i])

print(test)  # converted into numeric

test = ss.fit_transform(test)  # scaling the TestData

pred = NBClassifier.predict(test)
print("\nAPPLICANTS ELIGIBLE FOR LOAN FROM TEST DATA \n", pred)