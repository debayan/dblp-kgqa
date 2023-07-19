import streamlit as st
import streamlit.components.v1 as components
import torch
import os
import re
import json
import requests
import pandas as pd
import numpy as np
import time
import configparser
from transformers import (
        T5ForConditionalGeneration,
        T5Tokenizer,
        BartForConditionalGeneration,
        BartTokenizer
)
# Import utility functions
import sys
sys.path.insert(0, './utils/')
from utils import *

config = configparser.ConfigParser()
config.read('config.ini')
st.set_page_config(layout="wide")

if "num_combos" not in st.session_state:
    st.session_state.num_combos = 0
if "question" not in st.session_state:
    st.session_state.question = ''
if "results" not in st.session_state:
    st.session_state.results = []
if "clear" not in st.session_state:
    st.session_state.clear = 0
if "refresh" not in st.session_state:
    st.session_state.refresh = 0
if "combo_names" not in st.session_state:
    st.session_state.combo_names = {}
if "entered_ques" not in st.session_state:
    st.session_state.entered_ques = "Which papers did Yann LeCun write?"
if "submit_ques" not in st.session_state:
    st.session_state.submit_ques = False

# Load the pre-trained T5 model and tokenizer
def setup_model(model_name, batch_size, epoch):
    if re.match(r"t5*", model_name):
        print("\nModel type: T5\n")
        model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict = True)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    else:
        print("\nModel type: BART\n")
        model = BartForConditionalGeneration.from_pretrained("facebook/"+model_name, return_dict = True)
        tokenizer = BartTokenizer.from_pretrained("facebook/"+model_name)

    output_dir = './models/'
    load_model(output_dir, model, 'model_{}.pth'.format(model_name))
    return model, tokenizer

@st.cache_data
def infer(question, _model, _tokenizer, device):
    # Tokenize
    input_question = _tokenizer.encode_plus(question, padding=True, truncation=True, return_tensors='pt')
    input_question = input_question.to(device)
    # Generate answer containing label and type
    output = _model.generate(input_ids=input_question['input_ids'], attention_mask=input_question['attention_mask'], max_length=512)
    predicted = [_tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in output]
    return predicted

def separate_ents(predicted):
    ents = predicted.split("[SEP]")
    return ents[:-1]

def get_labels_types(ents):
    labels = []
    types =[]
    for ent in ents:
        all_types = []
        if ":" in ent:
            i = ent.rindex(":")
            labels.append(ent[:i].strip())
            temp = ent[i+1:]
            all_types = temp.split(',')
            all_types = [item.strip() for item in all_types[:-1]]
            types.append(all_types)
    print("label and types: ")
    print(labels)
    print(types)
    return labels, types 

def predict_labels_types(question, model, tokenizer, device):
    predicted = infer(question, model, tokenizer, device)
    print("\n\npredicted")
    print(predicted)
    ents = separate_ents(predicted[0])
    labels, types = get_labels_types(ents)
    return labels, types 

def print_labels_types(labels, types):
    labels_df = pd.DataFrame(list(zip(labels, types)), columns=["Predicted Label", "Types"])
    st.markdown("###### Predicted Labels") 
    st.dataframe(labels_df, hide_index=True)

def make_clickable(link, strip=True):
    link, text = link.split(':::')
    if strip:
        link = link[1:-1]
    return f'<a target="_blank" href="{link}">{text}</a>'

@st.cache_data
def add_result(question, embedding, labels, types, _config, _environment):
    url = _config[_environment]['entitylinker'] + embedding
    headers = {"Content-Type": "application/json"}
    idx = 0
    responses = []
    for label, typs in zip(labels, types):
        data = {"question": question, "label": label, "type": typs}
        resp = requests.post(url, headers=headers, data=json.dumps(data))
        print(resp)
        resp_json = resp.json()
        responses.append(resp_json)
    result = {"labels":labels, "types":types, "responses":responses}
    return result

def update_results(result):
    st.session_state.results.append(result)
    st.session_state.num_combos += 1

def types_to_str(typs):
    typs_str = ''
    for t in typs:
        t = "https://dbpedia.org/ontology/"+t+":::"+t
        typs_str = typs_str + make_clickable(t, False) + " "
    return typs_str

def display_table(res):
    idx = 0
    df_final = pd.DataFrame(columns=("Label", "Types", "Distance", "Entity"))
    for label, typs in zip(res["labels"], res["types"]):
        resp_json = res["responses"][idx]
        dist = round(resp_json[0][0], 2)
        link = resp_json[0][1][0]
        ent_label = resp_json[0][1][1]
        link_str = link + ":::" + ent_label
        types_str = types_to_str(typs)
        obj = {"Label": ent_label, "Types": types_str, "Distance": dist, "Entity": link_str}
        df = pd.DataFrame(obj, columns=("Label", "Types", "Distance", "Entity"), index=[0])
        df_final = pd.concat([df_final, df], ignore_index=True)
        idx = idx + 1
    df_final['Entity'] = df_final['Entity'].apply(make_clickable)
    st.write(df_final.to_html(escape = False, index=False, justify='center'), unsafe_allow_html = True)

def display_res(res):
    idx = 0
    for label, typs in zip(res["labels"], res["types"]):
        resp_json = res["responses"][idx]
        dists = [item[0] for item in resp_json]
        links = [item[1][0] for item in resp_json]
        ent_labels = [item[1][1] for item in resp_json]
        st.markdown("**Label " + str(idx) + "**: " + label)
        link_strs = []
        for i in range(len(links)):
            link_str = links[i] + ":::" + ent_labels[i]
            link_strs.append(link_str)
        df = pd.DataFrame((list(zip(dists, link_strs))), columns=("Distance", "Entity"))
        df['Entity'] = df['Entity'].apply(make_clickable)
        st.write(df.to_html(escape = False), unsafe_allow_html = True)
        st.divider()
        idx = idx + 1

def refresh():
    st.session_state.refresh = 1

def clear_callback():
    st.session_state.results = []
    st.session_state.num_combos = 0
    st.session_state.combo_names = {} 
    refresh()

def newq_callback():
    clear_callback()
    st.session_state.selection_1 = "Use entered question"
    st.session_state.entered_ques = ''
    st.session_state.question = ''

def del_combo(del_model):
    idx = st.session_state.combo_names[del_model]
    st.session_state.results.pop(idx)
    st.session_state.num_combos -= 1
    st.session_state.combo_names.pop(del_model)
    st.session_state.refresh = 1

def use_entered_ques():
    st.session_state.selection_1 = "Use entered question"
    st.session_state.submit_ques = True
    st.session_state.refresh = 1

def use_selected_ques():
    st.session_state.entered_ques = ""
    st.session_state.submit_ques = True

def main(config, environment='staging'):
    st.header('DBLP Entity Linker')

    # Custom style
    st.markdown("""
    <style type="text/css">
    div[data-testid="stHorizontalBlock"]{
    border:10px;
    padding:5px;
    border-radius:10px;
    background:#f9f9f9;
    }
    div[class="block-container css-z5fcl4 e1g8pov64"]{
    padding:3rem;
    }
    table {
    word-wrap: break-all; 
    flex:1;
    }
    td {
    text-align: left;
    word-break:break-word;
    }
    th {
    
    }
    div[class="st-c7 st-cl st-cm st-ae st-af st-ag st-ah st-ai st-aj st-cn st-co"]{
    font-size:smaller;
    }
    .Aligner {
    display: flex;
    align-items: center;
    justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.question == '':
        disable_ques = False
    else:
        disable_ques = True
    
    # Mapping names
    model_name_display = {"T5-small":"t5-small", "T5-base":"t5-base"}
    embed_name_display = {"TransE":"transe", "ComplEx":"complex", "DistMult":"distmult"}

    # If all combos already there, disable selections
    if st.session_state.num_combos == len(model_name_display)*len(embed_name_display):
        disable_selection = True
    else:
        disable_selection = False
    
    # Question
    col0_ques, col2_newq = st.columns([0.8, 0.2])
    with col0_ques:
        entered_ques = st.text_input('Enter question', disabled=disable_ques, on_change=use_entered_ques, key="entered_ques")
    with col2_newq:
        if st.session_state.question != '':
            st.markdown('<div class="Aligner"', unsafe_allow_html=True)
            newq_btn = st.button('New question', type='primary',on_click=newq_callback)
            st.markdown('</div>', unsafe_allow_html=True)
    sample_ques = st.selectbox('Choose from samples', ["Use entered question","Between 'Graphics and Security: Exploring Visual Biometrics' and 'OOElala: order-of-evaluation based alias analysis for compiler optimization', which one was published first?","Which one has more number of authors, 'A Flexible and Generic Gaussian Sampler With Power Side-Channel Countermeasures for Quantum-Secure Internet of Things' or 'GEONGrid portal: design and implementations'?"], disabled=disable_ques, index=0, key="selection_1", on_change=use_selected_ques)
    
    input_form = st.sidebar.form('Form1')
    model_name = ''
    embedding = ''
    with input_form: 
        # Input
        st.markdown('###### 1. Label Predictor Model')
        model_name = st.radio('Select Model', options=model_name_display.keys(), label_visibility = 'collapsed', horizontal=False, disabled=disable_selection)  
        st.markdown('###### 2. Entity Ranker Embeddings')
        embedding = st.radio('Select Embeddings',embed_name_display.keys(), label_visibility='collapsed', horizontal=False, disabled=disable_selection)  
        submit_btn = st.form_submit_button('Submit', type="primary", disabled=disable_selection)

        # Load Model
        epochs = 50
        if model_name == 'T5-small':
            batch_size = 16
        else:
            batch_size = 4
        model, tokenizer = setup_model(model_name_display[model_name], batch_size, epochs)
        device = 'cpu'
        model.to(device)
    
    # Display question and buttons
    del_btn = False
    if st.session_state.question != '':
        st.markdown('<p style="font-weight:bold;margin-bottom:-1rem;padding-top:1rem;padding-left:2rem;padding-right:2rem;">Question: </p>', unsafe_allow_html=True)
        st.markdown('<p style="padding-top:-1rem;padding-bottom:1rem;padding-left:2rem;padding-right:2rem;">'+st.session_state.question+' </p>', unsafe_allow_html=True)

        delete_form = st.sidebar.form('Form2')
        with delete_form:
            del_list = ['Select combination']
            del_list.extend(list(st.session_state.combo_names.keys()))
            del_list.append('All')
            del_model = st.selectbox('', options=del_list, label_visibility='collapsed', index=0)
            if st.session_state.num_combos == 0:
                disable_delete = True
            else:
                disable_delete = False
            del_btn = st.form_submit_button('Delete', type='primary', disabled=disable_delete)
    
    # Delete combination
    if del_btn:
        if del_model == 'Select combination':
            with st.sidebar:
                st.error("Please select a combination")
        elif del_model == 'All':
            clear_callback()
        else:
            del_combo(del_model)

    # Print the old results in two columns
    col1_out, col2_out = st.columns([0.5, 0.5])
    for i in range(st.session_state.num_combos):
        combo_name = list(st.session_state.combo_names)[i]
        if i % 2 == 0:
            with col1_out:
                with st.expander(combo_name, expanded=True):
                    display_table(st.session_state.results[i])
                with st.expander('Ranked Entities', expanded=False):
                    display_res(st.session_state.results[i])
        else:
            with col2_out:
                with st.expander(combo_name, expanded=True):
                    display_table(st.session_state.results[i])
                with st.expander('Ranked Entities', expanded=False):
                    display_res(st.session_state.results[i])
    # Add new combo if submitted
    if submit_btn or st.session_state.submit_ques:
        st.session_state.refresh = 1
        st.session_state.submit_ques = False
        if st.session_state.question != '' or len(entered_ques) and sample_ques == "Use entered question" or sample_ques != "Use entered question":
            disable_ques = True

            if st.session_state.question != '':
                question = st.session_state.question
            elif sample_ques == "Use entered question":
                question = entered_ques
            else:
                question = sample_ques
            st.session_state.question = question

            if model_name == '' and embedding == '':
                model_name = 'T5-small'
                embedding = 'TransE'

            combo_name = model_name + " + " + embedding
            if combo_name in st.session_state.combo_names:
                st.error("This combination already exists. Please try another.")
            else:
                st.session_state.combo_names[combo_name] = st.session_state.num_combos
                if st.session_state.num_combos % 2 == 0:
                    with col1_out:
                        with st.spinner('Predicting...'):
                            labels, types = predict_labels_types(question, model, tokenizer, device)
                        with st.spinner('Ranking...'):
                            result = add_result(question, embed_name_display[embedding], labels, types, config, environment)
                            update_results(result)
                        with st.expander(combo_name, expanded=True):
                            display_table(st.session_state.results[st.session_state.num_combos-1])
                        with st.expander('Ranked Entities', expanded=False):
                            display_res(st.session_state.results[st.session_state.num_combos-1])
                else:
                    with col2_out:
                        with st.spinner('Predicting...'):
                            labels, types = predict_labels_types(question, model, tokenizer, device)
                        with st.spinner('Ranking...'):
                            result = add_result(question, embed_name_display[embedding], labels, types, config, environment)
                            update_results(result)
                        with st.expander(combo_name, expanded=True):
                            display_table(st.session_state.results[st.session_state.num_combos-1])
                        with st.expander('Ranked Entities', expanded=False):
                            display_res(st.session_state.results[st.session_state.num_combos-1])
        else:
            st.error('Please enter or choose a question')
    
    if st.session_state.refresh==1:
        st.session_state.refresh = 0
        time.sleep(3)
        st.experimental_rerun()     
    print(st.session_state)

if __name__ == '__main__':
    if sys.argv[1] == 'production':
        main(config, 'production')
    else:
        main(config, 'staging')
