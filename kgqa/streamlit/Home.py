import streamlit as st
import requests
import sys
import pandas as pd
import json

st.set_page_config(layout="wide")

host = sys.argv[1]
default_questions = ["What is the Wikidata ID of Pablo G. Tahoces?", "Which publications did Loukmen Regainia write?", "The author Sameh S. Askar is primarily affiliated to which institution?", "List the venues in which Matt Rowe published papers in the last six years and the titles of these papers.", "Who are the co-authors of Kolosov, Kirill?", "Find the venue of the paper 'Consensus in non-commutative spaces' authored by Alain Sarlette."]

def process_entity(entid):
    try:
        url = 'https://dblp-kg.ltdemos.informatik.uni-hamburg.de/sparql'
        query = '''
                     SELECT distinct ?label where { 
                                                 %s <http://www.w3.org/2000/01/rdf-schema#label> ?label .
                     } 
                '''%(entid)
        #print(query)
        headers = {'Accept':'application/sparql-results+json'}
        r = requests.get(url, headers=headers, params={'format': 'json', 'query': query})
        json_format = r.json()
        #print(json_format)
        results = json_format['results']['bindings'][0]['label']['value']
        return results
    except Exception as err:
        print(err)
        return ''



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
        question = st.text_input("Ask a question:", value='Which papers did Debayan Banerjee publish?')
        submitted = st.form_submit_button("Submit")
        if submitted:
            response = infer(question)
            resp = response['output'][0]
            if resp['entities']:
                columns = ['Entity','Label']
                df = pd.DataFrame(columns=columns)
                entity_labels = [process_entity(x) for x in resp['entities']]
                for entity_label,entity_id in zip(entity_labels, resp['entities']):
                    row_data  = pd.DataFrame([{'Entity': entity_id[1:-1] , 'Label':entity_label}])
                    df = pd.concat([df,row_data], ignore_index=True)
                st.write("**Entities:**")
                st.markdown(df.to_html(render_links=True), unsafe_allow_html=True)
            else:
                st.write("No entities detected")
            if resp['query']:
                st.markdown('#')
                st.write("**SPARQL Query:**")
                st.write(resp['query'])
            if resp['answer']:
                st.write("**Answer:**")
                try:
                    answer_labels = []
                    for x in resp['answer']:
                        for key in x.keys():
                            if x[key]['type'] == 'uri':
                                if 'dblp' in x[key]['value']:
                                    label = process_entity('<'+x[key]['value']+'>')
                                    answer_labels.append((x[key]['value'],label))
                    if answer_labels:
                        columns = ['Entity','Label']
                        df = pd.DataFrame(columns=columns)
                        for url,label in answer_labels:
                            row_data  = pd.DataFrame([{'Entity': url , 'Label': label}])
                            df = pd.concat([df,row_data], ignore_index=True)
                        st.markdown(df.to_html(render_links=True), unsafe_allow_html=True)
                        st.write("**Answer JSON:**")
                        st.json(resp['answer'], expanded=False)
                    else:
                        st.write("**Answer JSON:**")
                        st.json(resp['answer'])
                except Exception as err:
                    print(err)
                    st.write("**Answer JSON:**")
                    st.json(resp['answer'])
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
                columns = ['Entity','Label']
                df = pd.DataFrame(columns=columns)
                entity_labels = [process_entity(x) for x in resp['entities']]
                for entity_label,entity_id in zip(entity_labels, resp['entities']):
                    row_data  = pd.DataFrame([{'Entity': entity_id[1:-1] , 'Label':entity_label}])
                    df = pd.concat([df,row_data], ignore_index=True)
                st.write("**Entities:**")
                st.markdown(df.to_html(render_links=True), unsafe_allow_html=True)
            else:
                st.write("No entities detected")
            if resp['query']:
                st.markdown('#')
                st.write("**SPARQL Query:**")
                st.write(resp['query'])
            if resp['answer']:
                st.write("**Answer:**")
                try:
                    answer_labels = []
                    for x in resp['answer']:
                        for key in x.keys():
                            if x[key]['type'] == 'uri':
                                if 'dblp' in x[key]['value']:
                                    label = process_entity('<'+x[key]['value']+'>')
                                    answer_labels.append((x[key]['value'],label))
                    if answer_labels:
                        columns = ['Entity','Label']
                        df = pd.DataFrame(columns=columns)
                        for url,label in answer_labels:
                            row_data  = pd.DataFrame([{'Entity': url , 'Label': label}])
                            df = pd.concat([df,row_data], ignore_index=True)
                        st.markdown(df.to_html(render_links=True), unsafe_allow_html=True)
                        st.write("**Answer JSON:**")
                        st.json(resp['answer'], expanded=False)
                    else:
                        st.write("**Answer JSON:**")
                        st.json(resp['answer'])
                except Exception as err:
                    print(err)
                    st.write("**Answer JSON:**")
                    st.json(resp['answer'])
            else:
                st.write("No answer could be fetched")


       

