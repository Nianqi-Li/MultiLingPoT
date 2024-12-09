import llm_classifier
from tqdm import tqdm
import math
import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r',encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data
  
def voting(input_list):
    if len(input_list) == 0:
        return None
    counts = {}
    for element in input_list:
        if element in counts:
            counts[element] += 1
        else:
            counts[element] = 1
    max_count = max(counts.values())
    modes = [key for key, value in counts.items() if value == max_count]
    if len(modes) == 1:
        return modes[0]
    else:
        return None

def classifier(query_results:dict):
    choose_type = llama_classifier.classifier(query_results['python']['question'], {key: query_results[key]['solution'] for key in query_results})
    return query_results[choose_type]['answer']

def get_result(query_results:dict):
    voting_result = voting([item['answer'] for item in query_results.values() if item['answer'] is not None])
    if voting_result is not None:
        return voting_result
    else:
        return classifier(query_results)
    
if __name__ == '__main__':
    testset = read_jsonl('testset_path.json')
    result_set = {"python": read_jsonl('python_result.jsonl'),
                  "c++": read_jsonl('c_result.jsonl'),
                  "java": read_jsonl('java_result.jsonl'),
                  "matlab": read_jsonl('matlab_result.jsonl')}
    correct = 0
    for i in tqdm(range(len(testset))):
        answer = get_result({"python":result_set["python"][i], "c++":result_set["c++"][i], "java":result_set["java"][i], "matlab":result_set["matlab"][i]})
        if answer is not None and math.fabs(answer - float(testset[i]["answer"])) < 1e-2:
            correct += 1
    print(correct/len(testset))
