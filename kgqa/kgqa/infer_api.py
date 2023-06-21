# You can also adapt this script on your own summarization task. Pointers for this are left as comments.
import numpy as np

from numpy.linalg import norm
import re
import argparse
import json
import base64
import logging
import math
import os
import random
from pathlib import Path
import sys
import requests
import datasets
import nltk
import numpy as np
import configparser
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import Counter
import transformers
from filelock import FileLock
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)
from flask_cors import CORS, cross_origin
from transformers.utils import get_full_repo_name, is_offline_mode
from transformers.utils.versions import require_version
import itertools

from flask import Flask, jsonify, request

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

logging.getLogger("requests").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )

    args = parser.parse_args()
    # Sanity checks

    return args

def sparqlquery(query):
    try:
        url =  'https://dblp-kg.ltdemos.informatik.uni-hamburg.de/sparql'
        print(query)
        headers = {'Accept':'application/sparql-results+json'}
        r = requests.get(url, headers=headers, params={'format': 'json', 'query': query})
        json_format = r.json()
        print(json_format)
        results = json_format['results']['bindings']
        return results
    except Exception as err:
        print(err)
        return ''




def resolveentity(label, enttype):
    try:
        url = 'https://dblp-kg.ltdemos.informatik.uni-hamburg.de/sparql'
        query = '''
                     SELECT distinct ?x where { 
                                                 ?x <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> %s .
                                                 ?x <http://www.w3.org/2000/01/rdf-schema#label> "%s" .
                     } 
                '''%(enttype,label)
        print(query)
        headers = {'Accept':'application/sparql-results+json'}
        r = requests.get(url, headers=headers, params={'format': 'json', 'query': query})
        json_format = r.json()
        print(json_format)
        results = json_format['results']['bindings'][0]['x']['value']
        return results
    except Exception as err:
        print(err)
        return '' 

def infer(question):
    args = parse_args()

    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    logger.setLevel(logging.INFO)

    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
        #tokenizer.add_tokens(['{','}','select','where','?vr0','?vr1','?vr2','?vr3','?vr4','?vr5','?vr6'], special_tokens=True )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        #tokenizer.add_tokens(['{','}','select','where','?vr0','?vr1','?vr2','?vr3','?vr4','?vr5','?vr6'], special_tokens=True )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        ).to(device)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""


    # Temporarily set max_target_length for training.
    max_target_length = 512
    padding = "do_not_pad"


    def preprocess_function():
        inputs = [question]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=512, padding=padding, truncation=True)
        return model_inputs


    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]

        return preds

    model.eval()

    gen_kwargs = {
        "max_length": 512,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_beams
    }
    with torch.no_grad():
        input = preprocess_function()
        generated_tokens = model.generate(
            torch.tensor(input['input_ids']).to(device).long(),
            attention_mask=torch.tensor(input['attention_mask']).to(device).long(),
            **gen_kwargs,
        )
        generated_tokens = generated_tokens.cpu().numpy()

        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        print(decoded_preds)
        f = lambda A, n=3: [A[i:i+n] for i in range(0, len(A), args.num_beams)]
        beamed_preds = f(decoded_preds)
        print(beamed_preds)
        original_inputs = tokenizer.batch_decode(input["input_ids"], skip_special_tokens=True)
        print(original_inputs)
        nonempty = False
        beamoutputs = [] 
        for beams,original_input in zip(beamed_preds,original_inputs):
            beamitem = {}
            queryresult = []
            for beam in beams:
                print(beam)
                pred = beam
                pred = pred.replace('?answer',' ?answer').replace(' WHERE',' WHERE {').replace(' ent>',' <ent>').replace(' /ent>',' </ent>').replace('prefix@@','<https://')+'}'
                entlabels = re.findall( r'<ent> (.*?) </ent>', pred)
                for entlabel in entlabels:
                    label,enttype = entlabel.split(' : ')
                    print(label,enttype)
                    ent = resolveentity(label,enttype)
                    if not ent:
                        print("entity could not be linked for ",entlabel)
                    else:
                        print("linked entity is ",ent," for ",entlabel)
                        print("replacing ...")
                        print(pred)
                        pred = pred.replace('<ent> '+entlabel+' </ent>','<'+ent+'>')
                        print(pred)
                response = sparqlquery(pred)
                print("response: ",response)
                beamoutputs.append({'query':pred,'answer':response})
        return beamoutputs





app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/answer', methods=['POST'])
@cross_origin()
def answer():
    data = request.get_json()
    print(data)
    question = data['question']
    output_str = infer(question)
    return jsonify({'output': output_str})

if __name__ == '__main__':
    app.run(host="0.0.0.0")
