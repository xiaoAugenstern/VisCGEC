import json

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def out_put(path,target_path):
    infos = []
    target_data = read_json(target_path)
    for item in target_data:
        info = {
            'id':item['id'],
            'img_id':item['img_id'],
            'source_text':item['source_ground_truth'],
            'target_text':item['target_text']
        }
        infos.append(info)
    with open(path, 'w') as f:
        json.dump(infos, f,ensure_ascii=False, indent=4)


def check(input_path):
    data = read_json(input_path)
    for item in data:
        len_sgt = len(item['source_ground_truth'])
        len_source = len(item['source'])
        len_target = len(item['target'])
        # 计算长度差值
        diff_sgt_source = abs(len_sgt - len_source)
        diff_sgt_target = abs(len_sgt - len_target)
        diff_source_target = abs(len_source - len_target)

        # 检查长度差值是否超过3
        if diff_sgt_source ==1 :
            print(f"Item with ID: {item.get('id', 'N/A')}")
            print('img_id:',item['img_id'])
            print(f"source_ground_truth length: {len_sgt}")
            print(f"source_text length: {len_source}")
            print(f"target_text length: {len_target}")
            print('source_ground_truth:', item['source_ground_truth'])
            print('source_text:', item['source'])
            print('target_ground_truth:', item['target'])
            print(
                f"Differences: sgt-source: {diff_sgt_source}, sgt-target: {diff_sgt_target}, source-target: {diff_source_target}")
            print("-" * 40)


def parse_annotations(annotation_str):
    """
    Parse the annotation string to extract error positions and their type (e.g., 'S', 'R', 'M', or 'noop').
    Returns a list of tuples, where each tuple is (start_pos, end_pos, type).
    """
    annotations = []
    lines = annotation_str.strip().split('\n')
    for line in lines:
        if line.startswith('A'):
            parts = line.split('|||')
            pos = parts[0].split()
            start_pos = int(pos[1])
            end_pos = int(pos[2])
            error_type = parts[1]  # Error type (e.g., M, R, S, or noop)
            annotations.append((start_pos, end_pos, error_type))
    return annotations

def calculate_metrics(annotations_target, annotations_predict):
    """
    Calculate precision, recall, and F0.5 based on the comparison between
    target annotations and predict annotations.
    """
    # Extract positions from target and predict annotations

    def get_position(annatations):
        positions = set()
        for start, end, operation in annatations:
            if operation != 'noop':
                if operation == 'M':
                    positions.add(start)
                else:
                    positions.update(range(start, end))
        return positions


    target_positions = get_position(annotations_target)
    predict_positions = get_position(annotations_predict)

    # True Positives (TP): Correctly predicted error positions
    TP = len(target_positions & predict_positions)
    print('TP:',TP)

    # False Positives (FP): Predicted error positions that are not in target
    FP = len(predict_positions - target_positions)
    print('FP:',FP)

    # False Negatives (FN): Real error positions that are missed by prediction
    FN = len(target_positions - predict_positions)
    print('FN:',FN)

    # Calculate precision, recall, and F0.5
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    beta = 0.5
    f0_5 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if precision + recall > 0 else 0

    return precision, recall, f0_5




def format_json(input_path, output_path):
    # 读取原始文件内容
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read()

    try:
        # 尝试解析 JSON 内容
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        return

    # 将数据写入新的 JSON 文件，同时格式化输出
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)



    def update(path):
        data = read_json(path)
        infos = []
        for item in data:
            id = item['id']
            pos = False
            for target_item in viscgec:
                if target_item['id'] == id:
                    pos = True
                    info = {
                        'id':item['id'],
                        'img_id':item['img_id'],
                        'source_text':item['source_text'],

                        'target_text':target_item['target'],
                        'source_ground_truth':target_item['source_ground_truth'],
                    }
                    infos.append(info)
                    continue
            print(f'{id} {pos}')

        with open(path, 'w', encoding='utf-8') as file:
            json.dump(infos, file, ensure_ascii=False, indent=4)
        print(f'{path} successfully updated')


def compare_datasets(ocr_data, clip_data,text_data):
    # 用于存储不同项的信息
    differences = []

    # 创建一个字典，以便快速查找
    ocr_dict = {item['id']: item for item in ocr_data}
    clip_dict = {item['id']: item for item in clip_data}
    text_data = {item['id']: item for item in text_data}

    # 比较两个数据集中的每个项目
    for id, ocr_item in ocr_dict.items():
        if id in clip_dict:
            clip_item = clip_dict[id]
            text_item = text_data[id]
            # 比较 source_ground_truth
            if ocr_item['source_ground_truth'] != clip_item['source_ground_truth']\
                    or ocr_item['source_ground_truth'] != text_item['source_text']:
                differences.append({
                    'id': id,
                    'field': 'source_ground_truth',
                    'ocr_value': ocr_item['source_ground_truth'],
                    'clip_value': clip_item['source_ground_truth']
                })

            # 比较 target_text
            if ocr_item['target_text'] != clip_item['target_text'] \
                    or ocr_item['target_text'] != text_item['target_text']:
                differences.append({
                    'id': id,
                    'field': 'target_text',
                    'ocr_value': ocr_item['target_text'],
                    'clip_value': clip_item['target_text']
                })
        else:
            differences.append({
                'id': id,
                'field': 'missing_in_clip',
                'ocr_value': ocr_item
            })

    # 检查在 OCR 数据集中不存在但在 CLIP 数据集中存在的 ID
    for id, clip_item in clip_dict.items():
        if id not in ocr_dict:
            differences.append({
                'id': id,
                'field': 'missing_in_ocr',
                'clip_value': clip_item
            })

    print('differencs:',differences)
    return differences



if __name__ == '__main__':

    viscgec_path= '../dataset/VisCGEC.json'
    viscgec = read_json(viscgec_path)

    visual_ocr_train_path = '../dataset/Visual_OCR/trainset.json'
    visual_ocr_test_path = '../dataset/Visual_OCR/testset.json'
    visual_ocr_valid_path = '../dataset/Visual_OCR/validset.json'
    visual_ocr_train = read_json(visual_ocr_train_path)
    visual_ocr_test = read_json(visual_ocr_test_path)
    visual_ocr_valid = read_json(visual_ocr_valid_path)
    print('visual_ocr_train',len(visual_ocr_train))
    print('visual_ocr_test',len(visual_ocr_test))
    print('visual_ocr_valid',len(visual_ocr_valid))

    visual_text_train_path = '../dataset/Visual_text/trainset.json'
    visual_text_test_path = '../dataset/Visual_text/testset.json'
    visual_text_valid_path = '../dataset/Visual_text/validset.json'
    visual_text_train = read_json(visual_text_train_path)
    visual_text_test = read_json(visual_text_test_path)
    visual_text_valid = read_json(visual_text_valid_path)
    print('visual_text_train',len(visual_text_train))
    print('visual_text_test',len(visual_text_test))
    print('visual_text_valid',len(visual_text_valid))

    visual_yolo_train_path = '../dataset/Visual_YOLO_CLIP/trainset.json'
    visual_yolo_test_path = '../dataset/Visual_YOLO_CLIP/testset.json'
    visual_yolo_valid_path = '../dataset/Visual_YOLO_CLIP/validset.json'
    visual_yolo_train = read_json(visual_yolo_train_path)
    visual_yolo_test = read_json(visual_yolo_test_path)
    visual_yolo_valid = read_json(visual_yolo_valid_path)

    print('visual_yolo_train',len(visual_yolo_train))
    print('visual_yolo_test',len(visual_yolo_test))
    print('visual_yolo_valid',len(visual_yolo_valid))

    ''' compare datasets'''
    compare_datasets(visual_ocr_valid, visual_yolo_valid, visual_text_valid)
    compare_datasets(visual_ocr_test, visual_yolo_test, visual_text_test)
    compare_datasets(visual_ocr_train, visual_yolo_train, visual_text_train)

    with open('predicts.json','r',encoding='utf-8') as f:
        predicts = json.load(f)

    total = 0
    support = 0
    against = 0
    for item in predicts:
        if item.get('a'):
            total += 1
            if item['a'] == 1:
                support += 1
            elif item['a'] == 0:
                against += 1
    print('total',total)
    print('support',support)
    print('against',against)
    print('support ratio',support/total)
