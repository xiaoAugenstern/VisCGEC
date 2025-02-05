import json
predict_path = '/home/xiaoman/project/gec/HandwrittenGEC/gectoolkit/llm/LLaMA-Factory/saves/qwen2-7b-visual-text-grammar/lora/predict/generated_predictions.jsonl'
from collections import Counter

# 初始化计数器
tp = 0  # True Positives
fp = 0  # False Positives
fn = 0  # False Negatives
# 初始化计数器
label_counter = Counter()
predict_counter = Counter()

# 遍历数据
with open(predict_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        label = data['label']
        predict = data['predict']
        prompt = data['prompt']

        # 统计频次
        label_counter[label] += 1
        predict_counter[predict] += 1

        # 计算 TP, FP, FN
        if label == predict:
            tp += 1  # 预测正确
        else:
            fp += 1  # 预测错误
            fn += 1  # 标签不匹配
            print('---------------------------------------------------')
            print(prompt)
            print('正确的label：',label)
            print('预测的label：',predict)
            print('---------------------------------------------------\n\n')



# 计算 Precision, Recall, F1
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f'precision: {precision}, recall: {recall}, f1: {f1}')
# 输出统计结果
print("Label frequency count:")
for label, count in label_counter.items():
    print(f"Label {label}: {count} times")

print("\nPredict frequency count:")
for predict, count in predict_counter.items():
    print(f"Predict {predict}: {count} times")