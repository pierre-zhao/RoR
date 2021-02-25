import argparse
import json

from eval_quac import f1_score,normalize_answer
import numpy as np
import math

def LCS(a,b):
    a = normalize_answer(a)
    b = normalize_answer(b)
    A = a.split(' ') 
    B = b.split(' ')
    if len(A) <= 4:
        if ' '.join(A) in b:
            return True
        else:
            return False
    elif len(B) <= 4:
        if ' '.join(B) in a:
            return True
        else:
            return False
    else:
        for i in range(len(A)-4): 
            if ' '.join(A[i:i+4]) in b: 
                return True 
    return False

def add_arguments(parser):
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)
    parser.add_argument("--answer_threshold", help="threshold of answer", required=False, default=0.1, type=float)

def convert_quac(input_file,
                 output_file,
                 answer_threshold):
    with open(input_file, "r") as file:
        input_data = json.load(file)
    data_lookup = {}
    
    with open('/home/user31/notespace/zhaojing/ReAnswer/data/predict.transformer_one.detail.json','r') as f:
        input_data1 = json.load(f)
    
    spectial = []
    num = 0
    for data,data1 in zip(input_data,input_data1):
        qas_id = data["qas_id"]
        id_items = qas_id.split('#')
        id = id_items[0]
        turn_id = int(id_items[1])
        predictions = [i["predict_text"] for i in data["top_predicts"]]
        rank_score = [i["predict_score"] for i in data["top_predicts"]]
        predict_score = [i["predict_score"] for i in data["top_predicts"]]
         
        f1 = [f1_score(i,data["label_text"]) for i in predictions]
        for i,j,v in zip(data["top_predicts"],f1,predict_score):
          i["f1_score"] = j
          i["exp"] = math.exp(v)
       
        sort_rank = sorted(rank_score, reverse=True)
#         rank1 = rank_score.index(sort_rank[0])
        if rank_score[0] > 1.99:    # - sort_rank[-1])> 10 :# 5and (predict_score[0] - predict_score[rank1])/predict_score[rank1] < 0.1 :
            delta = 0.3
            final_score = [delta*i+(1-delta)*j for i,j in zip(rank_score,predict_score)]
            max_index = final_score.index(max(final_score))
          
        else:
            delta = 0.0
            final_score = [delta*i+(1-delta)*math.exp(j) for i,j in zip(rank_score,predict_score)]
            max_index = final_score.index(max(final_score))
        # max_index = rank_score.index(max(rank_score))
        if f1.index(max(f1)) == rank_score.index(max(rank_score)):# and min(f1)!= 0:
          num += 1
          spectial.append(data)
#         max_index = rank_score.index(max(rank_score))

        no_answer = 1.0*data["no_answer_score"] + 0.0*data1["no_answer_score"]
        yes_no_list = ["y", "x", "n"]
        yes_no = yes_no_list[data["yes_no_id"]]
        
        follow_up_list = ["y", "m", "n"]
        follow_up = follow_up_list[data["follow_up_id"]]
        
        if (no_answer >= answer_threshold and data['follow_up_probs'][0] < 0.7):
            answer_text = "CANNOTANSWER"
        else:
            answer_text = predictions[max_index]
#            answer_text = data["predict_text"]
        
        if id not in data_lookup:
            data_lookup[id] = []
        
        data_lookup[id].append({
            "qas_id": qas_id,
            "turn_id": turn_id,
            "answer_text": answer_text,
            "yes_no": yes_no,
            "follow_up": follow_up
        })
    print(num)
    with open('/home/user31/notespace/zhaojing/new.json',"w") as f:
      json.dump(spectial,f, indent=4)
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



