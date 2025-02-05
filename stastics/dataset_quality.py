import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from gectoolkit.evaluate.cherrant import parallel_to_m2, compare_m2_for_evaluation
import numpy as np

def read_json(file_path):
    """Read a JSON file and return its content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


class DatasetQuality:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name   # Visual_text
        self.dataset_path = f'../dataset/{self.dataset_name}'

        self.trainset_path = f'{self.dataset_path}/trainset.json'
        self.testset_path = f'{self.dataset_path}/testset.json'
        self.validset_path = f'{self.dataset_path}/validset.json'

        self.trainset_data = read_json(self.trainset_path)
        self.testset_data = read_json(self.testset_path)
        self.validset_data = read_json(self.validset_path)

        ''' average_length '''
        self.average_length(self.trainset_path)
        self.average_length(self.testset_path)
        self.average_length(self.validset_path)

        ''' get m2 '''
        self.get_m2(self.trainset_path,type='train')
        self.get_m2(self.testset_path,type='test')
        self.get_m2(self.validset_path,type='valid')

        ''' average edits '''
        self.average_edits(self.trainset_path,type='train')
        self.average_edits(self.testset_path,type='test')
        self.average_edits(self.validset_path,type='valid')

    def average_length(self,data_path):
        data = read_json(data_path)
        data_length = len(data)
        average_source_length = 0
        average_target_length = 0
        fake_all = 0

        for item in data:
            source = item['source_text']
            target = item['target_text']

            fake = source.count('x') + source.count('X')
            fake_all += fake

            len_source = len(source)
            len_target = len(target)
            average_source_length += len_source
            average_target_length += len_target

        average_source_length = average_source_length / data_length
        average_target_length = average_target_length / data_length

        fake_per_sent = fake_all / data_length
        print(f'------- {data_path} --------')
        print('length of data: ', data_length)
        print('Average source length:', average_source_length)
        print('Average target length:', average_target_length)
        print('Fake per sent:', fake_per_sent)
        print()
        return average_source_length, average_target_length

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
                annotations.append((start_pos, end_pos, error_type))
        return annotations


    def get_m2(self,data_path,type):
        data = read_json(data_path)
        if not os.path.abspath(self.dataset_name):
            os.mkdir(self.dataset_name)

        st = os.path.join(os.path.abspath(self.dataset_name), f"{type}_st.para")
        st_m2_char = os.path.join(os.path.abspath(self.dataset_name), f"{type}_st.m2.char")

        sources = []
        targets = []

        for item in data:
            source = item['source_text']
            target = item['target_text']
            sources.append(source)
            targets.append(target)

        with open(st, 'w', encoding='utf-8') as file:
            for source, target in zip(sources, targets):
                source_target = source + '\t' + target
                file.write(source_target + '\n')

        p2m_hyp_args = parallel_to_m2.Args(file=st, output=st_m2_char)
        parallel_to_m2.main(p2m_hyp_args)
        print(f'{st},{st_m2_char} successfully')

    def average_edits(self, data_path,type):
        data = read_json(data_path)
        data_length = len(data)
        st_m2_char = os.path.join(os.path.abspath(self.dataset_name), f"{type}_st.m2.char")

        with open(st_m2_char, 'r', encoding='utf-8') as file:
            annotation = file.read()
            annotations = [self.parse_annotations(s) for s in annotation.strip().split('\n\n')]

        edits_all = 0
        r_num = 0
        m_num = 0
        s_num = 0
        w_num = 0

        for annotation_list in annotations:
            edits_num = len(annotation_list)   # per sentence
            edits_all += edits_num

            for a in annotation_list:
                error_type = a[2]
                if error_type == 'M':
                    m_num += 1
                elif error_type == 'S':
                    s_num += 1
                elif error_type == 'W':
                    w_num += 1
                elif error_type == 'R':
                    r_num += 1

        edits_per_sent = edits_all / data_length
        Missing_per_sent = m_num / data_length
        Redundant_per_sent = r_num / data_length
        Substitution_per_sent = s_num / data_length
        Word_Order_per_sent = w_num / data_length

        print(f'------ {data_path} -----')
        print('edits_per_sent', edits_per_sent)
        print('Missing_per_sent', Missing_per_sent)
        print('Redundant_per_sent', Redundant_per_sent)
        print('Substitution_per_sent', Substitution_per_sent)
        print('Word_Order_per_sent', Word_Order_per_sent)
        print()


def draw():
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # plt.rc('font', family='Times New Roman')
    error_types = ['Missing', 'Redundant', 'Substitution', 'Word-Order']
    total_errors = [128, 83, 225, 20]
    correct_errors = [23, 14, 67, 2]

    uncorrect_errors = [total - correct for total, correct in zip(total_errors, correct_errors)]

    colors = ['#ffcccc', '#ffe0cc', '#ccf2cc', '#ccccff']


    plt.figure(figsize=(8, 6))
    bars = plt.bar(error_types, uncorrect_errors, color=colors, edgecolor='gray', linewidth=1)

    plt.title('Uncorrected Errors by Type', fontsize=22)
    plt.xlabel('Error Types', fontsize=20)
    plt.ylabel('Number of Uncorrected Errors', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.xticks(fontsize=18)
    plt.ylim(0, 170)

    # 显示每个柱状图上方的具体数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 1), ha='center', va='bottom', fontsize=18)

    # 添加图例来表示颜色含义
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=colors[0], label='Missing'),
        mpatches.Patch(color=colors[1], label='Redundant'),
        mpatches.Patch(color=colors[2], label='Substitution'),
        mpatches.Patch(color=colors[3], label='Word-Order')
    ]
    plt.legend(handles=legend_patches, fontsize=12)

    # 显示图表
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # method = 'Visual_text'
    # dataset_quality = DatasetQuality(method)
    # print(f'Method {method} dataset quality successfully!')

    draw()