import csv
from Bio import SeqIO
from transformers import GPT2Tokenizer, Seq2SeqTrainingArguments
from prot2text_dataset.torch_geometric_loader import Prot2TextDataset
from prot2text_model.utils import Prot2TextTrainer
from prot2text_model.Model import Prot2TextModel
from prot2text_model.tokenization_prot2text import Prot2TextTokenizer
import torch
import os

def extract_empty_second_column(filename):
    result = []
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
            
        # 跳过标题行（如果有）
        next(reader, None)
            
        for row in reader:
            # 确保行至少有2列
            if len(row) >= 2:
                # 检查第二列是否为空（去除前后空格后）
                if not row[1].strip():
                    result.append(row[0])   
    return result

def get_uniprot2string(datapath:str):
    uniprot2string = dict()
    with open(f"{datapath}") as f:
        for line in f:
            it = line.strip().split(' ')
            uniprot2string[it[0]] = it[1]
    return uniprot2string
# 使用函数
filename = "Biotexts_miss.csv"
datapath="uniprot2string.txt"
empty_second_column_items = extract_empty_second_column(filename)

# 打印结果
print("empty_second_column_items:",len(empty_second_column_items))

string2uniprot = {}
uniprot2string = get_uniprot2string(datapath)
for uniprotid in uniprot2string:
    string2uniprot[uniprot2string[uniprotid]] = uniprotid

seqs = dict()
for record in SeqIO.parse(f'network.fasta', 'fasta'):
    seqs[string2uniprot[str(record.id)]] = str(record.seq)

print("seqs:",len(seqs))    

model_path="./models/esm2text_base"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = Prot2TextTokenizer.from_pretrained(model_path)
model = Prot2TextModel.from_pretrained(model_path)

results = []
count=0
for protid in empty_second_column_items:
    descrpition = model.generate_protein_description(protein_pdbID=None,
                                                 protein_sequence=seqs[protid], 
                                                 tokenizer=tokenizer,
                                                 device=device)
    results.append({
                'Name': protid,
                'Function': descrpition
            })
    count=count+1
    print("count:",count)

print("results:",len(results)) 
 
csv_filename = 'protein_Biotext_generate_description.csv'

# 写入CSV文件
try:
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        # 获取字段名（字典的键）
        fieldnames = results[0].keys()
        
        # 创建CSV写入器
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 写入表头
        writer.writeheader()
        
        # 写入数据行
        writer.writerows(results)
    
    print(f"数据已成功写入到 {csv_filename}")
except Exception as e:
    print(f"写入CSV文件时出错: {e}")  