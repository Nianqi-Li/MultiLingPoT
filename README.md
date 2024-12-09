# MultiLingPoT: Enhancing Mathematical Reasoning with Multilingual Program Fine-tuning

Program-of-Thought (PoT), which aims to use programming language instead of natural language as an intermediate step in reasoning, is an important way for LLMs to solve mathematical problems.
By offloading computation to a code interpreter, LLMs can leverage their reasoning abilities while overcoming computational limitations.
However, current PoT research only focuses on single language PoT, ignoring the differences between different programming languages.
Therefore, this paper proposes an multilingual program reasoning method, MultiLingPoT.
This method allows the model to answer questions using multiple programming languages by fine-tuning on multilingual data.
Additionally, hybrid methods, categorized into prior and posterior, are employed, allowing the model to select the most suitable language for each problem.
Our experimental results show that the training of MultiLingPoT significantly improves each program's mathematical reasoning.
Moreover, with proper mixing, the performance of MultiLingPoT can be further improved, achieving a 6\% increase compared to the single-language PoT with the same amount of training.

## File Descriptions
### 1. Dataset 
- `gsm8k_multilingpot.json` and `math_multilingpot.json` are our multilingual PoT datasets constructed by ChatGPT based on the GSM8K and MATH, containing 26,359 and 14,775 PoT samples in Python, C++, Java, and Matlab.
- `gsm8k_pythonpot.json` and `math_pythonpot.json` are our augmented Python PoT datasets, containing 26,414 and 14,634 samples, ensuring consistency in the total training data between SinglePoT and MultiLingPoT.

### 2. Python Scripts
- `case_based_choice.py`: Using the results of similar cases in each language to choose the current language to use.
- `self_consistency.py`: Voting for the final result using the results of four languages.
- `small_model_scorer.py`: Using a small model (bert/codebert) to predict the performance of each language on the problem. Depending on whether code is used as an evaluation criterion, either `query-only` or `query-code` can be chosen.
- `llm_scorer.py`:  Using LLM (Llama3) to predict the performance of each language on the problem. Depending on whether code is used as an evaluation criterion, either `query-only` or `query-code` can be chosen.
- `vote_with_scorer.py`: Start by voting using self_consistency; in the case of a tie, using llm_scorer to score and select the result.

### 3. YAML Configuration
`multilingpot_train.yaml` and `multilingpot_inference.yaml` are the configuration files for training and inference using llama-factory. All our experiments follow the parameters specified in these files.

## Usage
In order to train the MultiLingPoT model, it is preferred to ensure that Llama-Factory is installed:

For training, please use the following commands:
```bash
llamafactory-cli train multilingpot_train.yaml
```
For testing, please use the following command:
```bash
llamafactory-cli train multilingpot_inference.yaml
```
For using the different hybrid strategies in MultiLingPoT, please call the corresponding program in the `hybrid` folder directly.
