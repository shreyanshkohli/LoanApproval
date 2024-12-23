import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib

data1 = pd.read_csv('Python/Data/loan_sanction_train.csv')
data2 = pd.read_csv('Python/Data/loan_sanction_test.csv')
data = pd.concat([data1, data2], axis=0)

data = data.drop(['Loan_ID','Dependents'], axis=1)

data.fillna({
    'Gender': data.Gender.mode()[0],
    'Married': data.Married.mode()[0],
    'LoanAmount': data.LoanAmount.median(),
    'Self_Employed': data.Self_Employed.mode()[0],
    'Loan_Amount_Term': data.Loan_Amount_Term.mode()[0],
    'Credit_History': data.Credit_History.mode()[0],
    'Loan_Status': data.Loan_Status.mode()[0]
}, inplace=True)

ordinal = ['Married', 'Education', 'Self_Employed', 'Property_Area']
le = LabelEncoder()
for column in ordinal:
    data[column] = le.fit_transform(data[column])

nominal = ['Gender']
encodedData = pd.get_dummies(data[nominal], drop_first=True)
data = pd.concat([data, encodedData], axis=1)
data.drop(['Gender'], axis=1, inplace=True)

if data['Loan_Status'].dtype == 'object':
    data['Loan_Status'] = le.fit_transform(data['Loan_Status'])

x = data.drop('Loan_Status', axis=1)
y = data.Loan_Status

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=20)

# model1 = cross_val_score(LogisticRegression(max_iter=1000), xtrain, ytrain).mean()
# model2 = cross_val_score(SVC(), xtrain, ytrain).mean()
# model3 = cross_val_score(DecisionTreeClassifier(), xtrain, ytrain).mean()
# model4 = cross_val_score(RandomForestClassifier(), xtrain, ytrain).mean()

# print(model1, model2, model3, model4, sep='\n')

model = RandomForestClassifier(n_estimators=100)
model.fit(xtrain, ytrain)
p = model.predict(xtest)
score = model.score(xtest, ytest)
print(p, score)

# score: 83%