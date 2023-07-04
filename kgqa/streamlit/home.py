import streamlit as st
import requests
import sys

host = sys.argv[1]
default_questions = ["What is the Wikidata ID of Pablo G. Tahoces?", "Which publications did Loukmen Regainia write?", "The author Sameh S. Askar is primarily affiliated to which institution?", "List the venues in which Matt Rowe published papers in the last six years and the titles of these papers.", "Who are the co-authors of Kolosov, Kirill and where are they affiliated?", "Find the venue of the paper 'Consensus in non-commutative spaces' authored by Alain Sarlette."]

@st.cache_data
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


col1,col2 = st.columns([1,3])
with col2:
    with st.form("form1"):
        question = st.text_input("Ask a question:", value='How many papers does Chris Biemann have?')
        submitted = st.form_submit_button("Submit")
        if submitted:
            response = infer(question)
            resp = response['output'][0]
            if resp['entities']:
                st.write("Entities:")
                for ent in resp['entities']:
                    st.write(ent)
            else:
                st.write("No entities detected")
            if resp['query']:
                st.write("SPARQL Query:")
                st.write(resp['query'])
            if resp['answer']:
                st.write("Answer:")
                st.write(resp['answer'])
            else:
                st.write("No answer could be fetched")


 
buttons = []
with col1:
    for q in default_questions:
        x = st.button(q)
        buttons.append(x)
for idx,button in enumerate(buttons):
    if button:
        with col2:
            response = infer(default_questions[idx])
            resp = response['output'][0]
            if resp['entities']:
                st.write("Entities:")
                for ent in resp['entities']:
                    st.write(ent)
            else:
                st.write("No entities detected")
            if resp['query']:
                st.write("SPARQL Query:")
                st.write(resp['query'])
            if resp['answer']:
                st.write("Answer:")
                st.write(resp['answer'])
            else:
                st.write("No answer could be fetched")



       

