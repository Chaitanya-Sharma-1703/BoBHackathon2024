import numpy as np 
from flask import Flask, request
from predict import make_prediction
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
loan=pd.read_csv("./Loan_default.csv")
categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus', 
'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
label_encoder = LabelEncoder()
final={}
# Encoding categorical columns
for col in categorical_columns:
    label_encoder.fit_transform(loan[col])
    lenamemapping=dict(zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_)))
    loan[col]=label_encoder.fit_transform(loan[col])
    print(loan[col].unique())
    final[col]=lenamemapping

print(final)
def label_encoder_retriever(a,b):
    arr=["Age","Income","LoanAmount","CreditScore","MonthsEmployed","NumCreditLines","InterestRate","LoanTerm","DTIRatio",
    "Education","EmploymentType","MaritalStatus","HasMortgage","HasDependents","LoanPurpose","HasCoSigner"]
    res=final[arr[a]][b]
    return res

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to my Flask API</h1>"
        "</body>"
        "</html>"
    )
    return body

@app.route("/predict", methods=["POST"])
def predict():
    data_json = request.get_json()
    
    Age = data_json["Age"]
    Income = data_json["Income"]
    LoanAmount = data_json["LoanAmount"]
    CreditScore = data_json["CreditScore"]
    MonthsEmployed=data_json["MonthsEmployed"]
    NumCreditLines=data_json["NumCreditLines"]
    InterestRate=data_json["InterestRate"]
    LoanTerm=data_json["LoanTerm"]
    DTIRatio=data_json["DTIRatio"]
    Education=data_json["Education"]
    Education=label_encoder_retriever(9,Education)
    EmploymentType=data_json["EmploymentType"]
    EmploymentType=label_encoder_retriever(10,EmploymentType)
    MaritalStatus=data_json["MaritalStatus"]
    MaritalStatus=label_encoder_retriever(11,MaritalStatus)
    HasMortgage=data_json["HasMortgage"]
    HasMortgage=label_encoder_retriever(12,HasMortgage)
    HasDependents=data_json["HasDependents"]
    HasDependents=label_encoder_retriever(13,HasDependents)
    LoanPurpose=data_json["LoanPurpose"]
    LoanPurpose=label_encoder_retriever(14,LoanPurpose)
    HasCoSigner=data_json["HasCoSigner"]
    HasCoSigner=label_encoder_retriever(15,HasCoSigner)
    data = np.array([[Age,Income,LoanAmount,CreditScore,MonthsEmployed,NumCreditLines,
    InterestRate,LoanTerm,DTIRatio,Education,EmploymentType,
    MaritalStatus ,HasMortgage,HasDependents,LoanPurpose,HasCoSigner]])
    predictions = make_prediction(data)
    
    return str(predictions)

if __name__ == "__main__":
    app.run(debug=True)