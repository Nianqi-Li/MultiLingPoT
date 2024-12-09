import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llamafactory.data.template import TEMPLATES
from peft import PeftModel, PeftConfig
import argparse
import json
from tqdm import tqdm
import math
from openai import OpenAI

DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default='', type=str)
parser.add_argument("--adapter_path", default='', type=str)
parser.add_argument("--classifier_type", default='query-code', choices=['query-only','query-code'], type=str)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=DTYPES['float16'], trust_remote_code=True)
if args.adapter_path != '':
    model = PeftModel.from_pretrained(model, args.adapter_path)
model.to("cuda")
model.eval()

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def template(query: str, code: str, program_type: str):
    instruction = "You are a scorer. For the input math problem and {key} code, determine if it can be solved correctly using the code and output \"Yes\" if it can be solved and \"No\" otherwise.".format(key=program_type)
    if args.classifier_type == 'query-code':
        input = "Determine if the problem can be solved correctly by {key} code:\nQuestion: {question}\n{key} Code: {code}".format(key=program_type, question=query, code=code)
    elif args.classifier_type == 'query-only':
        input = "Determine if the problem can be solved correctly by {key}:\n{question}".format(key=program_type, question=query)
    messages =  [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input}
    ]
    return messages

def program_classifier(query: str, code: str, program_type: str):
    messages = template(query, code, program_type)
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = 'pt'
    ).to(model.device)
    terminators =[
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    output = model.generate(
        ids,
        max_new_tokens = 10,
        eos_token_id = terminators,
        temperature = 0.01,
        output_scores=True,
        return_dict_in_generate=True
    )
    logits = output.scores
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
    yes_token_logits = logits[0][0][yes_token_id]
    no_token_logits = logits[0][0][no_token_id]
    if torch.isinf(yes_token_logits):
        yes_token_logits = torch.tensor(0.0, device=yes_token_logits.device)
    if torch.isinf(no_token_logits):
        no_token_logits = torch.tensor(0.0, device=no_token_logits.device)
    return yes_token_logits - no_token_logits

def classifier(query: str, code: dict):
    program_score = {"python":0,"c++":0,"java":0,"matlab":0}
    for p in program_score.keys():
        program_score[p] = program_classifier(query, code[p], p)
    choosen_program = max(program_score, key=program_score.get)
    return choosen_program

if __name__ == '__main__':
    correct = 0
    testset = read_jsonl('testset_path.json')
    result_set = {"python": read_jsonl('python_rseult.jsonl'),
                  "c++": read_jsonl('c_result.jsonl'),
                  "java": read_jsonl('java_result.jsonl'),
                  "matlab": read_jsonl('matlab_result.jsonl')}
    choose_times = {"python":0,"c++":0,"java":0,"matlab":0}
    for i in tqdm(range(len(testset))):
        choice = classifier(testset[i]['question'], {"python":result_set["python"][i]['solution'], "c++":result_set["c++"][i]['solution'], "java":result_set["java"][i]['solution'], "matlab":result_set["matlab"][i]['solution']})
        choose_times[choice] += 1
        if result_set[choice][i]['answer'] is not None and math.fabs(result_set[choice][i]['answer'] - float(result_set[choice][i]["correct"])) < 1e-2:
            correct += 1
    print(correct/len(testset))
