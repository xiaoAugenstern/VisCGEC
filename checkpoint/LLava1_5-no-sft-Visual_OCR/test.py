import json

file_path = '../LLava1_5-sft-Visual_OCR/predicts.json'
with open(file_path,'r',encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    original_text = item['predict']
    item['predict'] = original_text.split('ASSISTANT: ')[-1]

# 将修改后的内容写回JSON文件
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)