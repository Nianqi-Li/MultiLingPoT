from transformers import AutoTokenizer, BertForSequenceClassification
import json
from tqdm import tqdm
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--classifier_type", required=True, choices=['query-only','query-code'], type=str)
args = parser.parse_args()

if args.classifier_type == 'query-only':
    judge_c = BertForSequenceClassification.from_pretrained("QueryScoring.Bert.c").to("cuda")
    judge_python = BertForSequenceClassification.from_pretrained("QueryScoring.Bert.python").to("cuda")
    judge_java = BertForSequenceClassification.from_pretrained("QueryScoring.Bert.java").to("cuda")
    judge_matlab = BertForSequenceClassification.from_pretrained("QueryScoring.Bert.matlab").to("cuda")
elif args.classifier_type == 'query-code':
    judge_c = BertForSequenceClassification.from_pretrained("QueryCodeScoring.Bert.c").to("cuda")
    judge_python = BertForSequenceClassification.from_pretrained("QueryCodeScoring.Bert.python").to("cuda")
    judge_java = BertForSequenceClassification.from_pretrained("QueryCodeScoring.Bert.java").to("cuda")
    judge_matlab = BertForSequenceClassification.from_pretrained("QueryCodeScoring.Bert.matlab").to("cuda")
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def template(query:str):
    encoding = tokenizer(
        query,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    ).to("cuda")
    return encoding

def classifier(query:str, code:dict):
    score = {"c++":0, "python":0, "java":0, "matlab":0}
    if args.classifier_type == 'query-only':
        score["c++"] = judge_c(**template("Query:{query}\n".format(query=query))).logits
        score["python"] = judge_python(**template("Query:{query}\n".format(query=query))).logits
        score["java"] = judge_java(**template("Query:{query}\n".format(query=query))).logits
        score["matlab"] = judge_matlab(**template("Query:{query}\n".format(query=query))).logits
    elif args.classifier_type == 'query-code':
        score["c++"] = judge_c(**template("Query:{query}\nSolution:{solution}\n".format(query=query,solution=code['c++']))).logits
        score["python"] = judge_python(**template("Query:{query}\nSolution:{solution}\n".format(query=query,solution=code['python']))).logits
        score["java"] = judge_java(**template("Query:{query}\nSolution:{solution}\n".format(query=query,solution=code['java']))).logits
        score["matlab"] = judge_matlab(**template("Query:{query}\nSolution:{solution}\n".format(query=query,solution=code['matlab']))).logits
    return max(score, key=score.get)

if __name__ == '__main__':
    correct = 0
    testset = read_jsonl('testset_path.jsonl')
    result_set = {"python": read_jsonl('python_result.jsonl'),
                  "c++": read_jsonl('c_result.jsonl'),
                  "java": read_jsonl('java_result.jsonl'),
                  "matlab": read_jsonl('matlab_result.jsonl')}
    choose_times = {"python":0,"c++":0,"java":0,"matlab":0}
    for i in tqdm(range(len(testset))):
        choice = classifier(testset[i]['input'], {"python":result_set["python"][i]['solution'], "c++":result_set["c++"][i]['solution'], "java":result_set["java"][i]['solution'], "matlab":result_set["matlab"][i]['solution']})
        choose_times[choice] += 1
        if result_set[choice][i]['answer'] is not None and math.fabs(result_set[choice][i]['answer'] - float(result_set[choice][i]["correct"])) < 1e-2:
            correct += 1
    print(correct/len(testset))
