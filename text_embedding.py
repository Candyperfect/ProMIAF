import torch
import numpy as np
import csv
from transformers import BertTokenizer, BertModel
import math
from tqdm import tqdm
"""
"""

def get_uniprot2string(datapath:str):
    uniprot2string = dict()
    with open(f"{datapath}") as f:
        for line in f:
            it = line.strip().split(' ')
            uniprot2string[it[0]] = it[1]
    return uniprot2string

def get_ppi_pid2index(datapath:str):
    pid2index = dict()
    with open(f"{datapath}") as f:
        for line in f:
            line = line.strip().split(' ')
            pid2index[line[0]] = int(line[1])   
    return pid2index

def read_csv_to_dict(filename):
    """读取无表头的CSV文件并返回字典（第一列为键，第二列为值）"""  
    data_dict = {}
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:  # 确保每行至少有两列数据
                protein_name = row[0].strip()  # 蛋白质名称作为键
                bio_text = row[1].strip()      # 生物文本信息作为值
                data_dict[protein_name] = bio_text
    return data_dict


dict1 = read_csv_to_dict("FinalBiotexts.csv")
print("dict number:",len(dict1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
tokenizer=BertTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
model=BertModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
model = model.to(device)

count=0
for pid in dict1:
    count=count+1

    text=dict1[pid]
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    length = 510
    step=math.ceil(input_ids.shape[1]/length)
    temp=[]
    if step==1:
        with torch.no_grad():
            emb=model(input_ids)
            #print("emb.pooler_output:",emb.pooler_output.cpu().shape)
        temp.append(emb.pooler_output.cpu())
    else:
        for i in range(step):
            if i< (step-1):
                with torch.no_grad():
                    emb=model(input_ids[:,i*length:(i+1)*length])
                temp.append(emb.pooler_output.cpu())
            if i==step-1:
                with torch.no_grad():
                    emb=model(input_ids[:,i*length:])
                temp.append(emb.pooler_output.cpu())
    pres = torch.stack(temp, dim=0).mean(dim=0)
    np.savetxt(f'text_feature/{pid}.txt',pres.detach().numpy().tolist())
    if count%100==0:
        print("count:",count)
    #exit()


ppi_pid2index = get_ppi_pid2index("ppi_pid2index.txt")
uniprot2string = get_uniprot2string("uniprot2string.txt")
feature_matrix = np.zeros((len(ppi_pid2index), 768))
pids = [file.split('.')[0] for file in os.listdir(f'text_feature/') if file.endswith('.txt')]
assert len(pids) == len(ppi_pid2index)
for pid in tqdm(pids, desc = 'get text embedding...'):
    try :
        feature_matrix[int(ppi_pid2index[uniprot2string[pid]])] = np.loadtxt(f'text_feature/{pid}.txt').astype(np.float32)
    except Exception as e:
        print(f'Pack {pid} text embedding  Error! {e}')
np.save(f'{datapath}/text_feature.npy', feature_matrix)