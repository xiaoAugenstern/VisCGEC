import os
from gectoolkit.evaluate.cherrant import parallel_to_m2, compare_m2_for_evaluation
import json
from collections import Counter
import editdistance
import pandas as pd

def read_json(file_path):
    """Read a JSON file and return its content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

def remove_spaces(text):
    return text.replace(" ", "")

def replace_punctuation(text):
    # 定义英文标点和对应的中文标点
    punctuation_mapping = {
        ',': '，',
        # '.': '。',
        '?': '？',
        '!': '！',
        ";":'；',
        ":":'：'
    }
    # 遍历映射表，替换文本中的标点符号
    for en, zh in punctuation_mapping.items():
        text = text.replace(en, zh)
    return text

def check_space(data):
    pos = True
    for index,i in enumerate(data):
        if i == '':
            pos = False
            print(f'{data} line {index} exist None')
    return pos

def compute_prf(tp,fp,fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    beta = 0.5
    f0_5 = (1 + beta ** 2) * (precision * recall) / (
                beta ** 2 * precision + recall) if precision + recall > 0 else 0
    return {
            'precision': precision,
            'recall': recall,
            'f05_score': f0_5
    }

def detection_tp_fp_fn(target_annot, predict_annot):
    def get_position(annotations):
        positions = set()
        for start, end, error_type,_ in annotations:
            if error_type != 'noop':
                if error_type == 'M':
                    positions.add(start)
                else:
                    positions.update(range(start, end))
        return positions
    target_positions = get_position(target_annot)
    predict_positions = get_position(predict_annot)
    TP = len(target_positions & predict_positions)  # True Positives
    FP = len(predict_positions - target_positions)  # False Positives
    FN = len(target_positions - predict_positions)  # False Negatives
    return TP, FP, FN


class VisGEC_Evaluation:
    def __init__(self,method,correction_model,F0_5):
        self.method = method                       # Visual_OCR,Visual_YOLO_CLIP, Visual_text, GPT,Mucgec-seq2edit.Mucgec-seq2seq
        self.correction_model = correction_model   # SynGEC, GECToR, Qwen2-7B-Instruct
        self.F0_5 = F0_5

        self.trainset_path = None
        self.testset_path = None
        self.valdataset_path = None

        self.correction_sources = []
        self.correction_targets = []
        self.correction_predicts = []

        self.recogntion_sources = []
        self.recogntion_targets = []

        self.annotations_predict = []
        self.annotations_target = []

        self.hyp = None            # source_predict
        self.ref = None            # source_target
        self.hyp_m2_char = None    # source_predict.m2
        self.ref_m2_char = None    # source_target.m2

        self.checkpoint_path = None
        self.predicts_path = None

        self.get_dataset()        # get [xxx]set_path,[xxx]set_data
        self.get_predicts_file()  # get checkpoint_path, predicts_path,correction_predicts, correction_targets,correction_sources

        # correction
        self.correction_char_level()
        self.correction_sent_level()

        self.get_annotations()     # get annotations_predict, annotations_target

        # detection
        self.detection_char_level()
        self.detection_sent_level()
        self.print_metrics()

        # recognition
        self.get_recognition()
        self.recognition_metrics()

        # error analysis
        self.statics_RMSW()


    def get_dataset(self):
        if self.method == "Visual_OCR" or self.method == "Visual_YOLO_CLIP" or self.method == "Visual_text":
            self.trainset_path = f'dataset/{self.method}/trainset.json'
            self.testset_path = f'dataset/{self.method}/testset.json'
            self.validset_path = f'dataset/{self.method}/validset.json'

            self.trainset_data = read_json(self.trainset_path)
            self.testset_data = read_json(self.testset_path)
            self.validset_data = read_json(self.validset_path)
        else:
            self.testset_path = f'dataset/Visual_text/testset.json'
            self.testset_data = read_json(self.testset_path)
            print(f'{self.method} method no trainset,testset,validset path , so use Visual_text/testset.json')

    def get_predicts_file(self):
        if self.method == 'GPT4':
            self.checkpoint_path = 'checkpoint/GPT4-Visual_image'
            self.predicts_path = os.path.join(self.checkpoint_path, 'predicts_process.json')
            predict_data = read_json(self.predicts_path)
            results = []
            for item in predict_data:
                img_id = item['img_id']
                for test_item in self.testset_data:
                    if img_id == test_item['img_id']:
                        info = {
                            'source': item['detection_result'],
                            'target': test_item['target_text'],
                            'predict': item['detection_result']
                        }
                        results.append(info)
                        self.correction_predicts.append(item['prediction'])
                        self.correction_sources.append(item['detection_result'])
                        self.correction_targets.append(test_item['target_text'])
            write_json(results, os.path.join(self.checkpoint_path, f'predicts.json'))

        elif self.method in ["Visual_OCR","Visual_YOLO_CLIP","Visual_text"]:
            self.checkpoint_path = f'checkpoint/{correction_model}-{method}'
            if not os.path.exists(self.checkpoint_path):
                print('checkpoint path not exist')

            if self.correction_model in ['SynGEC','GECToR','Transformer','T5']:
                self.predicts_path = os.path.join(self.checkpoint_path, f'predicts_F0.5={self.F0_5}.json')
                predict_data = read_json(self.predicts_path)  # predicts_F0.5=xxx.json
                for index, item in enumerate(predict_data):
                    rewrite = item['Rewrite']
                    rewrite = remove_spaces(replace_punctuation(rewrite))
                    self.correction_predicts.append(rewrite)
                    self.correction_targets.append(self.testset_data[index]['target_text'])
                    self.correction_sources.append(self.testset_data[index]['source_text'])

            elif self.correction_model in ['Qwen2-7B-Instruct','Qwen-VL-Chat','Qwen2.5-14B-Instruct','LLava1_5-sft']:
                self.predicts_path = os.path.join(self.checkpoint_path, 'generated_predictions.jsonl')
                results = []
                with open(self.predicts_path, 'r', encoding='utf-8') as file:  # generated_predictions.jsonl
                    for index, line in enumerate(file):
                        data = json.loads(line)
                        predict = data.get('predict')
                        predict = remove_spaces(predict)
                        info = {
                            'source': self.testset_data[index]['source_text'],
                            'target': self.testset_data[index]['target_text'],
                            'predict': predict
                        }
                        results.append(info)
                        self.correction_predicts.append(predict)
                        self.correction_targets.append(self.testset_data[index]['target_text'])
                        self.correction_sources.append(self.testset_data[index]['source_text'])
                write_json(results, os.path.join(self.checkpoint_path, f'predicts.json'))

            elif self.correction_model in ['Qwen2-7B-no-sft-Instruct','LLava1_5-no-sft']:
                self.predicts_path = os.path.join(self.checkpoint_path, 'predicts.json')
                predict_data = read_json(self.predicts_path)
                for item in predict_data:
                    self.correction_predicts.append(item['predict'])
                    self.correction_targets.append(item['target'])
                    self.correction_sources.append(item['source'])

        elif self.method in ["Mucgec-seq2edit","Mucgec-seq2seq"]:
            self.checkpoint_path = f'checkpoint/{self.method}'
            self.predicts_path = os.path.join(self.checkpoint_path, 'testset.output')
            results = []
            with open(self.predicts_path, 'r', encoding='utf-8') as file:
                for index, line in enumerate(file):
                    line = line.strip()
                    source = self.testset_data[index]['source_text']
                    target = self.testset_data[index]['target_text']
                    predict = remove_spaces(line)
                    info = {
                        'source': source,
                        'target': target,
                        'predict': predict
                    }
                    results.append(info)
                    self.correction_predicts.append(predict)
                    self.correction_sources.append(source)
                    self.correction_targets.append(target)
            write_json(results, os.path.join(self.checkpoint_path, f'predicts.json'))

        elif self.method == "FCGEC-stg":
            self.checkpoint_path = f'checkpoint/{self.method}'
            self.predicts_path = os.path.join(self.checkpoint_path, 'output.xlsx')
            df = pd.read_excel(self.predicts_path, engine='openpyxl')
            output_column = df['Output']
            for index, predict in enumerate(output_column):
                source = self.testset_data[index]['source_text']
                target = self.testset_data[index]['target_text']
                self.correction_predicts.append(remove_spaces(predict))
                self.correction_sources.append(source)
                self.correction_targets.append(target)
        else:
            print(f'{self.method} method no predicts,checkpoint path')

        assert len(self.correction_predicts) == len(self.correction_targets) == len(
            self.correction_predicts), "Inconsistency in the number of lines in the document"
        if self.correction_targets == [] or self.correction_sources == [] or self.correction_predicts == []:
            print('correction_sources/targets/predicts None, please check your predict file')

    def correction_sent_level(self):
        pred_tar_same = []
        pred_tar_same_path = os.path.join(self.checkpoint_path, f'pred_tar_same_f0.5={self.F0_5}.json')
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for target, prediction,source in zip(self.correction_targets, self.correction_predicts,self.correction_sources):
            if target == prediction:
                true_positives += 1
                info = {
                    'source :': source,
                    'predict:': prediction,
                    'target :': target
                }
                pred_tar_same.append(info)
            else:
                false_positives += 1
                false_negatives += 1

        # 将pred和target相同的，存回pred_tar_same_path
        write_json(pred_tar_same, pred_tar_same_path)
        self.correction_sent_metrics = compute_prf(true_positives,false_positives,false_negatives)

    def correction_char_level(self):
        if self.checkpoint_path == None:
            print('!!!!!!!!!! checkpoint_path is None,Please check method !!!!!!!!!')
            exit()

        self.hyp = os.path.join(self.checkpoint_path, "hyp.para")
        self.ref = os.path.join(self.checkpoint_path, "ref.para")

        with open(self.hyp, 'w', encoding='utf-8') as hyp_file, \
                open(self.ref, 'w', encoding='utf-8') as ref_file:
            for source, target, predict in zip(self.correction_sources, self.correction_targets, self.correction_predicts):
                source_predict = source + '\t' + predict
                source_target = source + '\t' + target
                hyp_file.write(source_predict + '\n')
                ref_file.write(source_target + '\n')

        # 将 hyp 和 ref 转成m2格式
        self.hyp_m2_char = os.path.join(self.checkpoint_path, "hyp.m2.char")
        self.ref_m2_char = os.path.join(self.checkpoint_path, "ref.m2.char")

        # 调用 parallel_to_m2 生成 m2 文件
        p2m_hyp_args = parallel_to_m2.Args(file=self.hyp, output=self.hyp_m2_char)
        parallel_to_m2.main(p2m_hyp_args)

        p2m_ref_args = parallel_to_m2.Args(file=self.ref, output=self.ref_m2_char)
        parallel_to_m2.main(p2m_ref_args)

        compare_args = compare_m2_for_evaluation.Args(hyp=self.hyp_m2_char, ref=self.ref_m2_char)
        TP, FP, FN, Prec, Rec, F = compare_m2_for_evaluation.main(compare_args)
        self.correction_char_metrics = {
            'precision': Prec,
            'recall': Rec,
            'f05_score': F
        }

    def parse_annotations(self,annotation_str):
        annotations = []
        lines = annotation_str.strip().split('\n')
        for line in lines:
            if line.startswith('A'):
                parts = line.split('|||')
                pos = parts[0].split()
                start_pos = int(pos[1])
                end_pos = int(pos[2])
                error_type = parts[1]  # Error type (e.g., M, R, S, W or noop)
                modify = parts[2]
                annotations.append((start_pos, end_pos, error_type,modify))
        return annotations

    def get_annotations(self):
        with open(self.hyp_m2_char, 'r', encoding='utf-8') as hyp_file:
            hyp_annotation = hyp_file.read()
            self.annotations_predict = [self.parse_annotations(s) for s in hyp_annotation.strip().split('\n\n')]
        with open(self.ref_m2_char, 'r', encoding='utf-8') as ref_file:
            ref_annotation = ref_file.read()
            self.annotations_target = [self.parse_annotations(s) for s in ref_annotation.strip().split('\n\n')]

    def detection_sent_level(self):
        total_TP = total_FP = total_FN = 0
        for target_annot, predict_annot in zip(self.annotations_target, self.annotations_predict):
            TP, FP, FN = detection_tp_fp_fn(target_annot, predict_annot)
            # Sentence-level evaluation
            if TP > 0 and FP == 0 and FN == 0:
                total_TP += 1  # Correctly predicted all positions (no false positives or false negatives)
            elif FP > 0:
                total_FP += 1  # At least one false positive in the predicted sentence
            elif FN > 0:
                total_FN += 1  # At least one false negative in the predicted sentence
        self.detection_sent_metrics = compute_prf(total_TP, total_FP, total_FN)

    def detection_char_level(self):
        total_TP = total_FP = total_FN = 0
        for target_annot, predict_annot in zip(self.annotations_target, self.annotations_predict):
            TP, FP, FN = detection_tp_fp_fn(target_annot, predict_annot)
            total_TP += TP
            total_FP += FP
            total_FN += FN
        self.detection_char_metrics = compute_prf(total_TP,total_FP,total_FN)


    def print_metrics(self):
        print(' ---- Correction sentence level metrics ----')
        print('sentence precision:', self.correction_sent_metrics['precision'])
        print('sentence recall:', self.correction_sent_metrics['recall'])
        print('sentence f05_score:', self.correction_sent_metrics['f05_score'])

        print(' ---- Correction char level metrics ----')
        print('character precision:', self.correction_char_metrics['precision'])
        print('character recall:', self.correction_char_metrics['recall'])
        print('character f0.5_score:', self.correction_char_metrics['f05_score'])

        print(' ---- Detection sentence level metrics ----')
        print('sentence precision:', self.detection_sent_metrics['precision'])
        print('sentence recall:', self.detection_sent_metrics['recall'])
        print('sentence f05_score:', self.detection_sent_metrics['f05_score'])

        print(' ---- Detection char level metrics ----')
        print('char precision:', self.detection_char_metrics['precision'])
        print('char recall:', self.detection_char_metrics['recall'])
        print('char f05_score:', self.detection_char_metrics['f05_score'])

    def get_recognition(self):
        if self.method == 'Visual_OCR' or self.method == 'Visual_YOLO_CLIP':
            for item in self.testset_data:
                source_text = item['source_text']
                source_ground_truth = item['source_ground_truth']
                self.recogntion_sources.append(source_text)
                self.recogntion_targets.append(source_ground_truth)
        else:
            print(f'Method {self.method} Not Support get_recognition')
        assert len(self.recogntion_sources) == len(self.recogntion_targets)

    def calculate_character_accuracy(self):
        correct_chars = 0
        total_chars = 0
        for pred, gt in zip(self.recogntion_sources, self.recogntion_targets):
            correct_chars += sum(c1 == c2 for c1, c2 in zip(pred, gt))
            total_chars += len(gt)
        return correct_chars / total_chars if total_chars > 0 else 0.0

    def calculate_cer(self):
        total_distance = 0
        total_chars = 0
        for pred, gt in zip(self.recogntion_sources, self.recogntion_targets):
            distance = editdistance.eval(pred, gt)
            total_distance += distance
            total_chars += len(gt)
        return total_distance / total_chars if total_chars > 0 else 0.0

    def calculate_precision_recall_f1(self):
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred, gt in zip(self.recogntion_sources, self.recogntion_targets):
            tp, fp, fn = self.calculate_tp_fp_fn(pred, gt)
            true_positives += tp
            false_positives += fp
            false_negatives += fn

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1_score

    def calculate_tp_fp_fn(self, pred, gt):
        # Calculate TP, FP, FN based on character-level comparison
        pred_counter = Counter(pred)
        gt_counter = Counter(gt)
        tp = sum(min(pred_counter[c], gt_counter[c]) for c in set(pred).intersection(set(gt)))
        fp = sum(pred_counter[c] - min(pred_counter[c], gt_counter[c]) for c in set(pred))
        fn = sum(gt_counter[c] - min(pred_counter[c], gt_counter[c]) for c in set(gt))
        return tp, fp, fn

    def recognition_metrics(self):
        if self.method == 'Visual_OCR' or self.method == 'Visual_YOLO_CLIP':
            char_accuracy = self.calculate_character_accuracy()
            cer = self.calculate_cer()
            precision, recall, f1_score = self.calculate_precision_recall_f1()
            print(f"Character Accuracy: {char_accuracy:.4f}")
            print(f"Character Error Rate (CER): {cer:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
        else:
            print(f'Method {self.method} Not Support recognition_metrics')

    def statics_RMSW(self):
        with open(self.hyp_m2_char, 'r', encoding='utf-8') as file:
            annotation = file.read()
            annotations_source_predicit = [self.parse_annotations(s) for s in annotation.strip().split('\n\n')]

        with open(self.ref_m2_char, 'r', encoding='utf-8') as file:
            annotation = file.read()
            annotations_source_target = [self.parse_annotations(s) for s in annotation.strip().split('\n\n')]

        total_missing = 0
        total_redundant = 0
        total_substitution = 0
        total_word_order = 0
        predict_missing = 0
        predict_redundant = 0
        predict_substitution = 0
        predict_word_order = 0

        correct_missing = 0
        correct_redundant = 0
        correct_substitution = 0
        correct_word_order = 0

        for asp_list,ast_list in zip(annotations_source_predicit, annotations_source_target):
            for ast in ast_list:
                error_type_ast = ast[2]
                if error_type_ast == 'M':
                    total_missing += 1
                elif error_type_ast == 'S':
                    total_substitution += 1
                elif error_type_ast == 'W':
                    total_word_order += 1
                elif error_type_ast == 'R':
                    total_redundant += 1

            for asp in asp_list:
                error_type_asp = asp[2]
                if error_type_asp == 'M':
                    predict_missing += 1
                elif error_type_asp == 'S':
                    predict_substitution += 1
                elif error_type_asp == 'W':
                    predict_word_order += 1
                elif error_type_asp == 'R':
                    predict_redundant += 1

            for asp in asp_list:
                for ast in ast_list:
                    if asp == ast:
                        error_type = asp[2]
                        if error_type == 'M':
                            correct_missing += 1
                        elif error_type == 'S':
                            correct_substitution +=1
                        elif error_type == 'W':
                            correct_word_order += 1
                        elif error_type == 'R':
                            correct_redundant += 1

        error_missing_ratio = 1- (correct_missing / total_missing)
        error_redundant_ratio = 1- (correct_redundant / total_redundant)
        error_substitution_ratio = 1- (correct_substitution / total_substitution)
        error_word_order_ratio = 1- (correct_word_order / total_word_order)

        print('------- Error Analysis ------')
        print('total_missing:', total_missing)
        print('total_redundant:', total_redundant)
        print('total_substitution:', total_substitution)
        print('total_word_order:', total_word_order,'\n')
        print('predict_missing:', predict_missing)
        print('predict_redundant:', predict_redundant)
        print('predict_substitution:', predict_substitution)
        print('predict_word_order:', predict_word_order,'\n')
        print('correct_missing:', correct_missing)
        print('correct_redundant:', correct_redundant)
        print('correct_substitution:', correct_substitution)
        print('correct_word_order:', correct_word_order,'\n')
        print('Error missing ratio:', error_missing_ratio)
        print('Error redundant ratio:', error_redundant_ratio)
        print('Error substitution ratio:', error_substitution_ratio)
        print('Error Word order ratio:', error_word_order_ratio)



if __name__ == '__main__':
    method = 'Visual_OCR'        # Visual_OCR,Visual_YOLO_CLIP, GPT4, Visual_text,Mucgec-seq2seq,Mucgec-seq2edit,FCGEC-stg
    correction_model = 'LLava1_5-no-sft'      # SynGEC, GECToR, Qwen2-7B-Instruct,Qwen2-7B-no-sft-Instruct,None,Transformer,Qwen2.5-14B-Instruct,LLava1_5-no-sft,LLava1_5-sft
    F0_5 = None

    print('------------ evaluation ---------')
    print('method:', method)
    print('correction_model:', correction_model)
    print('F0_5:', F0_5)
    print()

    evaluation = VisGEC_Evaluation(method=method,correction_model=correction_model,F0_5=F0_5)
