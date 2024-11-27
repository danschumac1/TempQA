import json

if __name__ == "__main__":
    # Load the dataset
    with open('data/datasets/MenatQA/MenatQA.json') as f:
        timeQA_test = json.load(f)

    disorder = 0
    counter = 0
    scope = 0
    for ids, example in enumerate(timeQA_test, start=1):
        query = example["updated_question"]
        gold_answer = example["updated_answer"]
        
        for x in example['context']:
            if x['updated_text'] != '':
                disorder += 1
                break
        if example['type'] in ["counterfactual"]:
            counter += 1
        if example['type'] in ["narrow", "expand", "granularity"]:
            scope += 1

    print('Scope:', scope * 2 + counter) # Original question, modified question, and counterfactual questions
    print('Order:', disorder) # All questions with disordered context
    print('Counterfactual:', counter) # just counterfactuals