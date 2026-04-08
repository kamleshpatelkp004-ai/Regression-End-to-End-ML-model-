import streamlit as st 
import pandas as pd 
from sklearn.linear_model import LogisticRegression

st.title("Pass/fail result with Probability")

df = pd.read_csv("result.csv")

model = LogisticRegression()
model.fit(df[["Hours"]], df["Result"])

hours = st.slider("study Hours", 0,10)

prob = model.predict_proba([[hours]])

st.write("fail probability",prob[0][0])
st.write("pass probability",prob[0][1])