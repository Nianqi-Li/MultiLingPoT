from openai import OpenAI
import os
import json
import numpy as np
import random
import math
from tqdm import tqdm

os.environ.update({"OPENAI_API_KEY": ""})
client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

def get_embedding(text, model=EMBEDDING_MODEL):
    return client.embeddings.create(input=text, model=model).data[0].embedding

def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

icl_examples = read_jsonl('icl_example.jsonl')

def get_similar_cases(input):
    embeddings = get_embedding(input)
    similar_events = sorted([
        (e, vector_similarity(e['embedding'], embeddings)) for e in icl_examples
    ], reverse=True, key=lambda x: x[1])
    return [e[0] for e in similar_events]

def case_based_choice(data):
    case_score = {"python": 0, "c++": 0, "java": 0, "matlab": 0}
    similar_cases = get_similar_cases(data['input'])
    for sc in similar_cases:
        for key in sc['correct']:
            case_score[key] += sc['correct'][key]
        if sum(case_score.values()) >= 10:
            break
    return max(case_score, key=case_score.get)

if __name__ == "__main__":
    correct = 0
    testset = read_jsonl('testset_path.jsonl')
    result_set = {"python": read_jsonl('trainset_python_result.jsonl'),
                  "c++": read_jsonl('trainset_c_result.jsonl'),
                  "java": read_jsonl('trainset_java_result.jsonl'),
                  "matlab": read_jsonl('trainset_matlab_result.jsonl')}
    for i in tqdm(range(len(testset))):
        choice = case_based_choice(testset[i])
        if result_set[choice][i]['answer'] is not None and math.fabs(result_set[choice][i]['answer'] - result_set[choice][i]["correct"]) < 1e-2:
            correct += 1
    print(correct/len(testset))
