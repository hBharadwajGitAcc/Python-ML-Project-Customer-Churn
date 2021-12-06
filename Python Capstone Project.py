import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os
print(os.getcwd())
os.chdir("C:\\Users\\user\\Downloads\\assignments\\python_projects")
os.getcwd()


# Read the file.
customer_churn = pd.read_csv("customer_churn.csv") 


# To display the first few rows.
customer_churn.head()


# To extract first observation, and take a look at it.
first_row = customer_churn.iloc[1,:] 


# To extract last observation, and take a look at it.
last_row = customer_churn.iloc[7042]


# To take a took at the Columns' Index values.
colmns = []
for col in customer_churn.columns:
    colmns.append(col)




# Extract the 5th column, and store it in customer_5.
customer_5 = customer_churn.iloc[:,4] 
customer_5.head()




# Extract the 15th column, and store it in customer_15.
customer_15 = customer_churn.iloc[:,14] 
customer_15.head()




# To extract male senior citizen whose payment method is electronic check.
senior_male_electronic = customer_churn[(customer_churn['gender']=='Male') & (customer_churn['SeniorCitizen']==1) & (customer_churn['PaymentMethod']=='Electronic check')]
senior_male_electronic.head(10)




# To extract customer whose tenure is > 70 or their monthly charges are >100.
customer_total_tenure = customer_churn[(customer_churn['tenure']>70) | (customer_churn['MonthlyCharges']>100)]
customer_total_tenure.head(10)




# To extract customers whose contract is 'two year', payment method is 'Mailed Check', and the value of Churn is 'Yes'.
two_mail_yes = customer_churn[(customer_churn['Contract']=='Two year') & (customer_churn['PaymentMethod']=='Mailed check') & (customer_churn['Churn']=='Yes')]
two_mail_yes




# To extract 333 random records.
customer_333 = customer_churn.sample(n=333)
customer_333.head()

len(customer_333)




# To get count of levels of Churn column.
customer_churn['Churn'].value_counts()




get_ipython().run_line_magic('matplotlib', 'inline')


# Bar-plot for 'InternetService' column
x=customer_churn['InternetService'].value_counts().keys().tolist()
y=customer_churn['InternetService'].value_counts().tolist()
plt.bar(x,y,color='orange')
plt.xlabel('Categories of Internet Service')
plt.ylabel('Count of Categories')
plt.title('Distribution of Internet Service')




# Histogram for 'tenure' column
plt.hist(customer_churn['tenure'],color='green',bins=30)
plt.xlabel('Tenure of the Customer')
plt.ylabel('Count')
plt.title('Distribution of tenure')




# Scatter-plot between Monthly Charges & tenure
plt.scatter(x=customer_churn['tenure'].head(20),y=customer_churn['MonthlyCharges'].head(20),color='brown')
plt.xlabel('Tenure of Customer')
plt.ylabel('Monthly Charges of Customer')
plt.title('Tenure vs Monthly Charges')
plt.grid(False)




# Box-plot between tenure & Contract using pandas
customer_churn.boxplot(column='tenure', by=['Contract'])
plt.grid(False)


# Box-plot between tenure & Contract using seaborn
import seaborn as sns
sns.boxplot(x='Contract', y='tenure', data=customer_churn, width=0.3)




# Simple Linear Regression
from sklearn import linear_model
from sklearn.model_selection import train_test_split


x = pd.DataFrame(customer_churn['tenure'])
y = customer_churn['MonthlyCharges']
 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# To build the model
from sklearn.linear_model import LinearRegression
SimpleLinearRegression_Model = LinearRegression()
SimpleLinearRegression_Model.fit(x_train,y_train)


# To do prediction
SimpleLinearRegression_Predictions = SimpleLinearRegression_Model.predict(x_test)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(SimpleLinearRegression_Predictions, y_test)
rmse = np.sqrt(mse)
rmse




# Simple Logistic Regression
x = pd.DataFrame(customer_churn['MonthlyCharges'])
y = customer_churn['Churn']


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.65,random_state=0)


from sklearn.linear_model import LogisticRegression
SimpleLogisticRegression_Model = LogisticRegression()
SimpleLogisticRegression_Model.fit(x_train,y_train)


SimpleLogisticRegression_Predictions = SimpleLogisticRegression_Model.predict(x_test)


from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(SimpleLogisticRegression_Predictions,y_test), accuracy_score(SimpleLogisticRegression_Predictions,y_test)


SimpleLogisticRegression_Confusion_Matrix = confusion_matrix(SimpleLogisticRegression_Predictions,y_test)
SimpleLogisticRegression_Confusion_Matrix
SimpleLogisticRegression_Accuracy_Score = accuracy_score(SimpleLogisticRegression_Predictions,y_test)
SimpleLogisticRegression_Accuracy_Score


# Binary Logistic Regression
x = pd.DataFrame(customer_churn.loc[:,['MonthlyCharges','tenure']])
y = customer_churn['Churn']


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.80,random_state=0)


from sklearn.linear_model import LogisticRegression
BinaryLogisticRegression_Model = LogisticRegression()
BinaryLogisticRegression_Model.fit(x_train,y_train)


BinaryLogisticRegression_Predictions = BinaryLogisticRegression_Model.predict(x_test)


from sklearn.metrics import confusion_matrix,accuracy_score    
print(confusion_matrix(BinaryLogisticRegression_Predictions,y_test),
      accuracy_score(BinaryLogisticRegression_Predictions,y_test))  


BinaryLogisticRegression_Confusion_Matrix = confusion_matrix(BinaryLogisticRegression_Predictions,y_test)
BinaryLogisticRegression_Confusion_Matrix
BinaryLogisticRegression_Accuracy_Score = accuracy_score(BinaryLogisticRegression_Predictions,y_test)
BinaryLogisticRegression_Accuracy_Score




# Decision Tree Classifier
x = pd.DataFrame(customer_churn['tenure'])
y = customer_churn['Churn']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)  


from sklearn.tree import DecisionTreeClassifier  
DecisionTreeClassifier_Model = DecisionTreeClassifier()  
DecisionTreeClassifier_Model.fit(x_train, y_train)  


DecisionTreeClassifier_Predictions = DecisionTreeClassifier_Model.predict(x_test)  


from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test, DecisionTreeClassifier_Predictions))   
print(accuracy_score(y_test, DecisionTreeClassifier_Predictions))  




# Random Forest Classifier
x = customer_churn[['tenure','MonthlyCharges']]
y = customer_churn['Churn']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)  


from sklearn.ensemble import RandomForestClassifier
RandomForestClassifier_Model =RandomForestClassifier(n_estimators=100)
RandomForestClassifier_Model.fit(x_train,y_train)


RandomForestClassifier_Predictions = RandomForestClassifier_Model.predict(x_test)


from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test, RandomForestClassifier_Predictions))   
print(accuracy_score(y_test, RandomForestClassifier_Predictions))  

