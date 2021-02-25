import argparse
import json
import math
import numpy as np
import os 

def add_arguments(parser):
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)
    parser.add_argument("--answer_threshold", help="threshold of answer", required=False, default=0.1, type=float)




import re
import string
from collections import Counter
def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
  prediction_tokens = normalize_answer(prediction).split()
  ground_truth_tokens = normalize_answer(ground_truth).split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def cross_f1_max(predictions):
    cross_f1_max = []
    for i in range(len(predictions)):
        index = list(range(len(predictions)))
        index.pop(i)
        cross_f1_max.append(max([f1_score(predictions[i], predictions[j]) for j in index]))
    return cross_f1_max
def cross_f1_mean(predictions):
    cross_f1_mean = []
    for i in range(len(predictions)):
        index = list(range(len(predictions)))
        index.pop(i)
        cross_f1_mean.append(sum([f1_score(predictions[i], predictions[j]) for j in index])/len(index))
    return cross_f1_mean

def convert_quac(input_file,
                 output_file,
                 answer_threshold):
    with open(input_file, "r") as file:
        input_data = json.load(file)
    
    data_lookup = {}
    for data in input_data:
        qas_id = data["qas_id"]
        id_items = qas_id.split('#')
        id = id_items[0]
        turn_id = int(id_items[1])
        
        yes_no_list = ["y", "x", "n"]
        yes_no = yes_no_list[data["yes_no_id"]]
        
        
        no_answer = data["no_answer_score"]
        follow_up_list = ["y", "m", "n"]
        follow_up = follow_up_list[data["follow_up_id"]]
        predictions = [i["predict_text"] for i in data["top_predicts"]]
        scores = [i["predict_score"] for i in data["top_predicts"]]
        cross_f1 = cross_f1_mean(predictions)
        delta = 0.6
#        final_score = [delta*i + (1-delta)*j for (i,j) in zip(scores,cross_f1)]
        final_score = [delta*math.exp(i) + (1-delta)*j for (i,j) in zip(scores,cross_f1)]
        max_index = final_score.index(max(final_score))
        
        if no_answer >= answer_threshold and data['follow_up_probs'][0] < 0.7: #and data['yes_no_probs'][0] < 0.6:
            answer_text = "CANNOTANSWER"
        else:
            #answer_text = data["predict_text"]
            answer_text = predictions[max_index]
        
        if id not in data_lookup:
            data_lookup[id] = []
        data_lookup[id].append({
            "qas_id": qas_id,
            "turn_id": turn_id,
            "answer_text": answer_text,
            "yes_no": yes_no,
            "follow_up": follow_up
        })
    
    with open(output_file, "w") as file:
        for id in data_lookup.keys():
            data_list = sorted(data_lookup[id], key=lambda x: x["turn_id"])
            
            output_data = json.dumps({
                "best_span_str": [data["answer_text"] for data in data_list],
                "qid": [data["qas_id"] for data in data_list],
                "yesno": [data["yes_no"] for data in data_list],
                "followup": [data["follow_up"] for data in data_list]
            })
            
            file.write("{0}\n".format(output_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    convert_quac(args.input_file, args.output_file, args.answer_threshold)
