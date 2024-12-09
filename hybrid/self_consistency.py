import random
import math
import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r',encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def get_resultset(file_paths):
    resultset = []
    for file_path in file_paths:
        resultset.append(read_jsonl(file_path))
    resultset = [list(row) for row in zip(*resultset)]
    return resultset

def find_mode(input_list):
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
    return random.choice(modes)

def self_consistency(results):
    correct = 0
    wrong = 0
    for question in results:
        ans = find_mode([res['answer'] for res in question if res['answer'] is not None])
        if ans is not None and math.fabs(ans - float(question[0]["correct"])) < 1e-2:
            correct += 1
        else:
            wrong += 1
    return correct / (correct + wrong)

if __name__ == "__main__":
    testset = 'testset_path.jsonl'
    path_files = [f'python_result.jsonl',
                  f'c_result.jsonl',
                  f'java_resultjsonl',
                  f'matlab_resultjsonl']
    resultset = get_resultset(path_files)
    sc = self_consistency(resultset)
    print(sc)
