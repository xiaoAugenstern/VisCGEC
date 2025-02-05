import json
import hashlib

with open('testset.json', 'r',encoding='utf-8') as f:
    data = json.load(f)

def generate_id(text):
    # 生成文本的 MD5 哈希值作为唯一 ID
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def transform_data(data, version="FCGEC EMNLP 2022"):
    transformed_data = {}
    for item in data:
        source_text = item["source_text"]
        sentence_id = generate_id(source_text)
        transformed_data[sentence_id] = {
            "sentence": source_text,
            "version": version
        }
    return transformed_data

transformed_data = transform_data(data)
with open('visgec_test.json', 'w',encoding='utf-8') as f:
    json.dump(transformed_data, f, ensure_ascii=False, indent=4)
    