import streamlit as st
import requests
import sys

host = sys.argv[1]

def infer(question):
    url = "http://%s:5000/answer"%(host)  # Replace with the actual API endpoint URL
    print(url)

    # Define the request payload
    payload = {
        'question': question
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        return response.json()  # Return the JSON response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

st.title("DBLP Scholarly Question Answering")

with st.form("form1"):
    question = st.text_input("Ask a question:", value='How many papers does Chris Biemann have?')
    submitted = st.form_submit_button("Submit")
    if submitted:
        response = infer(question)
        st.write(response)

    
    


