import joblib 
import pandas as pd

model = joblib.load("model.pkl")

def make_prediction(inputs): 
    """
    Make a prediction using the trained model 
    """
    inputs_df = pd.DataFrame(
        inputs, 
        columns=["Age", "Income", "LoanAmount", "CreditScore",
        "MonthsEmployed","NumCreditLines","InterestRate","LoanTerm",
        "DTIRatio","Education","EmploymentType",
        "MaritalStatus","HasMortgage","HasDependents","LoanPurpose","HasCoSigner"]
        )
    predictions = model.predict(inputs_df)
    
    return predictions