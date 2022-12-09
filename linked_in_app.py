import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import plotly.graph_objects as go
from PIL import Image

s = pd.read_csv("social_media_usage.csv")

def clean_sm(data):
    x = (np.where(data==1,1,0))
    return x



ss = pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]),
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] == 1, 1, 0),
    "married":np.where(s["marital"] == 1, 1, 0),
    "female":np.where(s["gender"] >= 3, np.nan,
                          np.where(s["gender"] == 2, 1, 0)),
    "age":np.where(s["age"] > 98, np.nan, s["age"])}).dropna().sort_values(by=["income","education"], ascending=True)


y = ss["sm_li"]
X = ss[["age", "education", "female", "income", "married", "parent"]]


# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility

lr = LogisticRegression(class_weight= "balanced")


# Fit algorithm to training data
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

confusion_matrix(y_test, y_pred)

pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap='RdYlBu')

print(classification_report(y_test, y_pred))

# In[263]:



st.image('LinkedIn_Image.png')
st.title("WELCOME")
st.header("This app is used to predict if someone is a LinkedIn user based on varying predictors.")

female = st.radio("Select Gender",["Female", "Male"])


if female == "Female": 
    female=1 
else:
    female=0

age=st.slider('Enter Age:')
#st.write(f"Age is {age}")


married =st.radio("Are you Married?",["Yes", "No"])                     

if married == "Yes": 
    married=1 
else:
    married =0

parent =st.radio("Do you have Children?",["Yes", "No"])  
                    

if parent == "Yes": 
    parent=1 
else:
    parent =0



education = st.selectbox("What is the Highest Level Of Education you have completed?", 
             options = ["Less than high school",
                        "High school incomplete",
                        "High school graduate",
                        "Some college, no degree",
                        "Two-year associate degree from a college or university",
                        "Four-year college or university degree/Bachelor's degree (e.g., BS, BA, AB)",
                        "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
                        "Postgraduate or professional degree, including master's, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)",
                         ])
#st.write(f"Highest Level Of Education Completed is {education}")

#st.write("**Convert Selection to Numeric Value**")

if education == "Less than high school":
   education = 1
elif education == "High school incomplete":
    education = 2
elif education == "High school graduate":
     education = 3
elif education == "Some college, no degree":
    education = 4
elif education == "Two-year associate degree from a college or university":
    education = 5
elif education == "Four-year college or university degree/Bachelor's degree (e.g., BS, BA, AB)":
    education = 6
elif education == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    education = 7
else:
    education = 8


income = st.selectbox("Select your Household Income level:", 
             options = ["Less than $10,000",
                        "$10,000 to $20,000",
                        "$20,000 to $30,000",
                        "$30,000 to $40,000",
                        "$40,000 to $50,000",
                        "$50,000 to $75,000",
                        "$75,000 to $100,000",
                        "$100,000 to $150,000",
                        "$150,000 or more"
                         ])
#st.write(f"Household Income is {income}")

#st.write("**Convert Selection to Numeric Value**")

if income == "Less than $10,000":
   income = 1
elif income == "$10,000 to $20,000":
    income = 2
elif income == "$20,000 to $30,000":
     income = 3
elif income == "$30,000 to $40,000":
    income = 4
elif income == "$40,000 to $50,000":
    income = 5
elif income == "$50,000 to $75,000":
    income = 6
elif income == "$75,000 to $100,000":
    income = 7
elif income == "$100,000 to $150,000":
    income= 8
else:
    income = 9


person = [age, education, female, income, married, parent]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

# Print predicted class and probability
#st.markdown(f"Predicted class: {predicted_class[0]}") # 0=not a user, 1= user

#st.markdown(f"Probability that this person is a LinkedIn User: {probs[0][1]}")
probability = round(probs[0][1]*100,2)

st.header(f"Probability of being a LinkedIn User: {probability}%")

if probability >= 50:
    isit = "Likely"
else: 
    isit = "Unlikely"


fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = probability,
    title = {'text': f"{isit} a LinkedIn User"},
    gauge = {"axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 50], "color":"#E2E2E2"},
                {"range": [50, 100], "color":"#1F77B4"}
            ],
            "bar":{"color":"white"}}
))

st.plotly_chart(fig)

st.markdown("The foundations of the app are based on Machine Learning principles and Logistic Regression.")
st.markdown("Produced by Abigail Hoffman under instruction from Dr. Lyon at Georgetown University.")
