import json 

with open('output/data/predict.best.every.feature.json','r') as f:
    every = json.load(f)

with open('quac/dev-quac.json','r') as f:
    quac_train = json.load(f)
    
every = [sum(i,[]) for i in every]


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def split_token(paragraph_text):
    doc_tokens = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
    return doc_tokens

passages = []
for data in quac_train['data']:
    for qas in data['paragraphs'][0]['qas']:
        passages.append(' '.join(split_token(data['paragraphs'][0]['context'])))

def is_overlapping(x1, x2, y1, y2):
    return max(x1, y1) <= min(x2, y2)        

def remove_overlap(positions):
    position_no_overlap = positions[:]
    all_no_overlap = 0
    for i in range(len(position_no_overlap)):
        for j in range(len(position_no_overlap)):
            if i != j and is_overlapping(position_no_overlap[i][0],position_no_overlap[i][1],position_no_overlap[j][0],position_no_overlap[j][1],):
                position_no_overlap.append((min(position_no_overlap[i][0],position_no_overlap[j][0]),max(position_no_overlap[i][1],position_no_overlap[j][1])))
                remove1 = position_no_overlap[i]
                remove2 = position_no_overlap[j]
                position_no_overlap.remove(remove1)
                position_no_overlap.remove(remove2)
                all_no_overlap = 1
                break
        if all_no_overlap == 1:
            break
    if all_no_overlap == 0 or len(positions) == 1:
        return positions
    else:
        return remove_overlap(position_no_overlap)
    
def answer_to_text(answer, source):
    start_position = source.index(answer[0])
    end_position = start_position + len(answer[0])
    position = [[start_position, end_position]]
    for ans in answer[1:]:
        start = source.index(ans)
        end = start + len(ans)
        match = 0
        for index in range(len(position)):
            if is_overlapping(start, end, position[index][0], position[index][1]):
                match =1
                if start < position[index][0] and end > position[index][1]:
                    position[index][0] = start
                    position[index][1] = end
                elif start < position[index][0] and end < position[index][1]:
                    position[index][0] = start                  
                elif start > position[index][0] and end > position[index][1]:
                    position[index][1] = end                  
                else :
                    pass
                break
            else:
                pass
        if match == 0:
            position.append([start,end])
    
    if len(position) == 1:
        position_no_overlap = position
    else:
        position_no_overlap = remove_overlap(position)
    
    span_text = [source[i:j+1] for (i,j) in position_no_overlap]
    return ' . '.join(span_text)

def find_lcsubstr(s1, s2): 
    s1 = s1.split(' ')
    s2 = s2.split(' ')
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    mmax=0
    p=0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                m[i+1][j+1]=m[i][j]+1
                if m[i+1][j+1]>mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return ' '.join(s1[p-mmax:p])


context = []
new_label = []
new_start = []
for e,p in zip(every, passages):
    e = [item["predict_text"] for item in e]
    an2text = answer_to_text(e,p)
    context.append(an2text)    
    new_label.append("CANNOTANSWER")
    new_start.append(-1)

tag = 0
for data in quac_train["data"]:
    for qas in data["paragraphs"][0]["qas"]:
        qas["context"] = context[tag]
        qas["orig_answer"]["text"] = new_label[tag]
        qas["orig_answer"]["answer_start"] = new_start[tag]
        tag += 1


with open('quac/quac.dev.answer.best.json','w') as f:
    json.dump(quac_train,f)  