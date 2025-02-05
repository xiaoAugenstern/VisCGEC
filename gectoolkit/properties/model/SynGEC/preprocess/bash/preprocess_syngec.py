import json
import os
from check import all_check
import subprocess
import os
import sys
from gectoolkit.properties.model.SynGEC.preprocess.utils import tokenization
from supar import Parser
import pickle
import torch
import copy


def json_to_txt(path, preprocess_path, dataset_name, type):
    # 构造预处理数据的路径
    dataset_dir = os.path.join(preprocess_path, dataset_name, type)

    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Directory created: {dataset_dir}")

    # 构造源文件和目标文件的路径
    src_path = os.path.join(preprocess_path, dataset_name, type, 'src.txt')
    tgt_path = os.path.join(preprocess_path, dataset_name, type, 'tgt.txt')

    # 读取 JSON 文件
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"JSON file loaded successfully from: {path}")
    except FileNotFoundError:
        print(f"Error: The file {path} does not exist.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file {path} is not a valid JSON file.")
        return

    # 写入源文件和目标文件
    try:
        with open(src_path, 'w', encoding='utf-8') as src_file, \
                open(tgt_path, 'w', encoding='utf-8') as tgt_file:
            for entry in data:
                source_text = entry['source_text']  # 错误的句子
                target_text = entry['target_text']  # 正确的句子

                # 写入source_text到src.txt，并以换行分割
                src_file.write(source_text + '\n')
                # 写入target_text到tgt.txt，并以换行分割
                tgt_file.write(target_text + '\n')

        print(f"Files written successfully to: {src_path} and {tgt_path}\n")
    except Exception as e:
        print(f"Error writing files: {e}")


def split(line):
    line = line.strip()
    line = line.replace(" ", "")
    line = tokenization.convert_to_unicode(line)
    if not line:
        return ''
    tokens = tokenizer.tokenize(line)
    return ' '.join(tokens)

# 处理文件的函数
def segment_char(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            tokenized_line = split(line)
            outfile.write(tokenized_line + '\n')


def run_gopar(in_file, out_file):
    try:
        command = f"CUDA_VISIBLE_DEVICES=6 python {parse_path} {in_file} {out_file} {gopar_path}"
        subprocess.run(command, shell=True, check=True)
        print(f"Step 4. Predict source-side trees {out_file} successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


def load(filename):
    """从文件加载句子列表。"""
    sents = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                res = line.rstrip().split()
                if res:
                    sents.append(res)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        raise
    return sents


def perform_dependency_parsing(input_file, output_file, model_path):
    """执行依赖解析并将结果写入文件。"""
    # 加载依赖解析器
    dep = Parser.load(model_path)
    # 读取输入文件
    input_sentences = load(input_file)
    # 进行依赖解析
    res = dep.predict(input_sentences, verbose=False, buckets=32, batch_size=3000, prob=True)
    # 保存解析结果和概率矩阵
    probs = []
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for r, t in zip(res, res.probs):
                f.write(str(r) + "\n")
                # 处理概率矩阵
                t1, t2 = t.split([1, len(t[0]) - 1], dim=-1)
                t = torch.cat((t2, t1), dim=-1)
                t = torch.cat((t, t.new_zeros((1, len(t[0])))))
                t.masked_fill_(torch.eye(len(t[0])) == 1.0, 1.0)
                t_list = t.numpy()
                probs.append(t_list)

        # 保存概率矩阵到文件
        with open(output_file + ".probs", "wb") as o:
            pickle.dump(probs, o)

        print(f"Dependency parsing results written to {output_file}.")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")
        raise


############################################################################################################
if __name__ == '__main__':

    dataset_name = 'Visual_YOLO_CLIP'

    base_project_path = '/home/xiaoman/project/gec/HandwrittenGEC'
    base_dataset_path = os.path.join(base_project_path, 'dataset')
    preprocess_path = os.path.join(base_project_path, 'gectoolkit/properties/model/SynGEC/preprocess/preprocess_data')
    parse_path = os.path.join(base_project_path, 'gectoolkit/model/SynGEC/src/src_gopar/parse.py')
    gopar_path = os.path.join(base_project_path, 'gectoolkit/properties/model/SynGEC/preprocess/gopar/biaffine-dep-electra-zh-gopar')
    vocab_file = os.path.join(base_project_path,'gectoolkit/properties/model/SynGEC/preprocess/dicts/chinese_vocab.txt')

    args_bart_path = os.path.join(base_project_path,f'gectoolkit/properties/model/SynGEC/args/Chinese/Chinese_{dataset_name}_bart.json')
    args_generate_path = os.path.join(base_project_path,f'gectoolkit/properties/model/SynGEC/args/Chinese/Chinese_{dataset_name}_generate.json')

    trainset_path = os.path.join(base_dataset_path, dataset_name, "trainset.json")
    testset_path = os.path.join(base_dataset_path, dataset_name, "testset.json")
    valset_path = os.path.join(base_dataset_path, dataset_name, "validset.json")


    ''' 1. 将数据集的json文件转化为txt文件 '''
    json_to_txt(path=trainset_path,preprocess_path=preprocess_path,dataset_name=dataset_name,type='trainset')
    json_to_txt(path=testset_path,preprocess_path=preprocess_path,dataset_name=dataset_name,type='testset')
    json_to_txt(path=valset_path,preprocess_path=preprocess_path,dataset_name=dataset_name,type='validset')

    ''' 2. check '''
    files = {
        'test_src': os.path.join(preprocess_path, dataset_name, 'testset', 'src.txt'),
        'test_tgt': os.path.join(preprocess_path, dataset_name, 'testset', 'tgt.txt'),
        'valid_src': os.path.join(preprocess_path, dataset_name, 'validset', 'src.txt'),
        'valid_tgt': os.path.join(preprocess_path, dataset_name, 'validset', 'tgt.txt'),
        'train_src': os.path.join(preprocess_path, dataset_name, 'trainset', 'src.txt'),
        'train_tgt': os.path.join(preprocess_path, dataset_name, 'trainset', 'tgt.txt'),
    }
    # 检查所有文件
    failed_files = []
    checks_passed = True
    for key, path in files.items():
        if all_check(file_path=path, out_path=path):
            print(f'\ncheck {path} successfully')
        else:
            print(f'\ncheck {path} failed')
            failed_files.append(path)
            checks_passed = False

    # 只有在所有检查都通过的情况下才执行下一步
    if checks_passed:
        print('\n--------------------------------------------------------------')
        print("step 2: All checks passed. Proceeding to the next step.")
    else:
        print('\n---------------------------------------------------------------')
        print(" The following files did not pass the check ")
        for file in failed_files:
            print(f" - {file}")
        print("Please check these files and try again.")
        exit(1)  # 退出程序

    ''' 3. 分句 '''
    train_src = os.path.join(preprocess_path, dataset_name, 'trainset', 'src.txt')
    train_tgt = os.path.join(preprocess_path, dataset_name, 'trainset', 'tgt.txt')
    valid_src = os.path.join(preprocess_path, dataset_name, 'validset', 'src.txt')
    valid_tgt = os.path.join(preprocess_path, dataset_name, 'validset', 'tgt.txt')
    test_src = os.path.join(preprocess_path, dataset_name, 'testset', 'src.txt')
    test_tgt = os.path.join(preprocess_path, dataset_name, 'testset', 'tgt.txt')

    train_src_char = os.path.join(preprocess_path, dataset_name, 'trainset', 'src.txt.char')
    train_tgt_char = os.path.join(preprocess_path, dataset_name, 'trainset', 'tgt.txt.char')
    valid_src_char = os.path.join(preprocess_path, dataset_name, 'validset', 'src.txt.char')
    valid_tgt_char = os.path.join(preprocess_path, dataset_name, 'validset', 'tgt.txt.char')
    test_src_char = os.path.join(preprocess_path, dataset_name, 'testset', 'src.txt.char')
    test_tgt_char = os.path.join(preprocess_path, dataset_name, 'testset', 'tgt.txt.char')

    # 初始化分词器
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    # 处理所有文件
    segment_char(train_src, train_src_char)
    segment_char(train_tgt, train_tgt_char)
    segment_char(valid_src, valid_src_char)
    segment_char(valid_tgt, valid_tgt_char)
    segment_char(test_src, test_src_char)
    segment_char(test_tgt, test_tgt_char)

    print(" step 3: All files segment char successfully.")


    ''' 4. 创建json文件 '''
    # example_bart_path = os.path.join(base_project_path,f'gectoolkit/properties/model/SynGEC/args/Chinese/Chinese_{dataset_name}_bart.json')
    # example_generate_path = os.path.join(base_project_path,f'gectoolkit/properties/model/SynGEC/args/Chinese/Chinese_{dataset_name}_generate.json')
    #
    # with open(example_bart_path, 'r', encoding='utf-8') as f:
    #     example_bart = json.load(f)
    #
    # args_bart = copy.deepcopy(example_bart_path)
    # args_generate = copy.deepcopy(example_generate_path)
    #
    # args_bart['data'] = f'./gectoolkit/properties/model/SynGEC/preprocess/preprocess_data/chinese_{dataset_name}_with_syntax_bart/bin'
    # args_bart['restore_file'] = f"./gectoolkit/properties/model/SynGEC/preprocess/gopar/chinese_{dataset_name}t_baseline/checkpoint_best.pt"
    # with open(args_bart_path, 'r', encoding='utf-8') as f:
    #     json.dump(args_bart, f, ensure_ascii=False, indent=4)
    #
    # args_generate['data'] = f'/gectoolkit/properties/model/SynGEC/preprocess/preprocess_data/chinese_{dataset_name}_with_syntax_bart/bin'
    # args_generate['conll_file'] = [
    #     f"./gectoolkit/properties/model/SynGEC/preprocess/preprocess_data/chinese_{dataset_name}_test_with_syntax_bart/bin/test.conll.src-tgt.src"
    # ],
    # args_generate['dpd_file'] = [
    #     f"./gectoolkit/properties/model/SynGEC/preprocess/preprocess_data/chinese_{dataset_name}_test_with_syntax_bart/bin/test.dpd.src-tgt.src"
    # ],
    # args_generate['probs_file'] = [
    #     f"./gectoolkit/properties/model/SynGEC/preprocess/preprocess_data/chinese_{dataset_name}_test_with_syntax_bart/bin/test.probs.src-tgt.src"
    # ],
    # args_generate['src_char_path'] = f"./gectoolkit/properties/model/SynGEC/preprocess/preprocess_data/chinese_{dataset_name}_test_with_syntax_bart/test.char.src"
    # args_generate['tgt_char_path'] = f"./gectoolkit/properties/model/SynGEC/preprocess/preprocess_data/chinese_{dataset_name}_test_with_syntax_bart/test.char.tgt"
    # with open(args_generate_path, 'r', encoding='utf-8') as f:
    #     json.dump(args_generate, f, ensure_ascii=False, indent=4)
    #
    # print(f'write to {args_bart_path} successfully.')
    # print(f'write to {args_generate_path} successfully.')
    # print(" step 4: args/chineses  bart_path  generate_path ")

