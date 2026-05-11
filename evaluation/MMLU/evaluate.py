import json
import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from lm_polygraph import estimate_uncertainty, WhiteboxModel
from lm_polygraph.estimators import AttentionScore, ClaimConditionedProbability, CocoaMSP, DegMat, Eccentricity, \
    EigenScore, EigValLaplacian, Focus, LexicalSimilarity, LUQ, MaximumSequenceProbability, MeanTokenEntropy, \
    Perplexity, SAR, SemanticDensity, SemanticEntropy
from lm_polygraph.utils.generation_parameters import GenerationParameters
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

STOP = []
SURE = []
UNSURE = []

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(input_list):
    prompt = input_list[0]
    k = len(input_list) - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], input_list[j+1])
    prompt += "\nAnswer:"
    return prompt

def format_shots(prompt_data):
    prompt = ""
    for data in prompt_data:
        prompt += data[0]
        k = len(data) - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], data[j+1])
        prompt += "\nAnswer:"
        prompt += data[k+1] + "\n\n"

    return prompt


def gen_prompt(input_list,subject,prompt_data):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    prompt += format_shots(prompt_data)
    prompt += format_example(input_list)
    return prompt

def inference(tokenizer,model,input_text,subject,prompt_data):
    full_input = gen_prompt(input_text,subject,prompt_data)
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    length = len(ids[0])     
    outputs = model.generate(
                ids,
                max_new_tokens = 1,
                output_scores = True,
                return_dict_in_generate=True
            )
    logits = outputs['scores'][0][0]    #The first token
    probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[0]],
                        logits[tokenizer("B").input_ids[0]],
                        logits[tokenizer("C").input_ids[0]],
                        logits[tokenizer("D").input_ids[0]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
    )
    output_text = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
    conf = np.max(probs)
        
    return output_text, full_input, conf.item()

def checksure(input_text):
    full_input = f"{input_text}. Are you sure you accurately answered the question based on your internal knowledge? I am"
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    outputs = model.generate(
                ids,
                max_new_tokens = 1,
                output_scores = True,
                return_dict_in_generate=True
            )
    logits = outputs['scores']
     #greedy decoding and calculate the confidence of sure and unsure
    pt = torch.softmax(torch.Tensor(logits[0][0]),dim=0)
    sure_prob = pt[SURE[0]]
    unsure_prob = pt[UNSURE[0]]
    sure_prob = sure_prob/(sure_prob+unsure_prob)   #normalization
       
    return sure_prob.item()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--domain', type=str, default="ID",choices=["ID","OOD"])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--result',type=str, default="MMLU")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",pad_token="</s>",add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto')

    gen_params = GenerationParameters(
        do_sample=False,
        max_new_tokens=80,
    )
    whitebox_model = WhiteboxModel(model, tokenizer, generation_parameters=gen_params)
    estimators = {
        "MSP": MaximumSequenceProbability(),
        "Token-Level Entropy": MeanTokenEntropy(),

        "Perplexity": Perplexity(),
        "Semantic Entropy": SemanticEntropy(),
        # "Focus": Focus(),  # FIXME: parameters
        "CCP": ClaimConditionedProbability(),
        "EigenScore": EigenScore(),
        "SAR": SAR(),
        "Semantic Density": SemanticDensity(),
        "Attention Score": AttentionScore(),

        "Lexical Similarity (Rouge-L)": LexicalSimilarity(),
        "LUQ": LUQ(),
        "DegMat": DegMat(),
        "Eccentricity": Eccentricity(),
        "EVL": EigValLaplacian(),
        "Cocoa MSP": CocoaMSP(),
    }
    uncertainty = defaultdict(list)

    STOP.append(tokenizer(".")['input_ids'][0])  #stop decoding when seeing '.'
    SURE.append(tokenizer("sure")['input_ids'][0])
    UNSURE.append(tokenizer("unsure")['input_ids'][0])

    results = [("prediction", "confidence", "certainty", *tuple(estimators.keys()))]
    data = {}
    prompt = {}
    with open(f"../../R-Tuning-data/MMLU/MMLU_{args.domain}_test.json",'r') as f:
        data = json.load(f)

    with open(f"../../R-Tuning-data/MMLU/MMLU_{args.domain}_prompt.json",'r') as f:
        prompt = json.load(f)
        
    for i in tqdm(data.keys()):  
        prompt_data = prompt[i]
        type_name = i
        for instance in tqdm(data[i]):
            output,full_input, predict_conf = inference(tokenizer,model,instance,i,prompt_data)
            sure_prob = checksure(f"{full_input}{output}")
            uncertainty_scores = {
                name: estimate_uncertainty(whitebox_model, estimator, input_text=full_input).uncertainty
                for name, estimator in estimators.items()
            }

            prediction = 1 if output == instance[1] else 0
            results.append((prediction, predict_conf, sure_prob, *tuple(uncertainty_scores.values())))

            if sure_prob > 0.5:
                uncertainty["sure"].append(uncertainty_scores)
            else:
                uncertainty["unsure"].append(uncertainty_scores)

        torch.cuda.empty_cache()
        
    os.makedirs("results",exist_ok=True)
    with open(f"results/{args.result}_{args.domain}.json",'w') as f:
        json.dump(results,f)
