import json
import os
import yaml
import logging
import time
from openai import OpenAI

def read_json(file_path):
    """Read a JSON file and return its content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


class LLM:
    def __init__(self,method,llm_model):
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)  # method
        project_root = os.path.dirname(current_dir)

        self.method = method                        # OCR, YOLO_CLIP
        self.name = f'Visual_{self.method}'         # Visual_OCR, Visual_YOLO_CLIP
        self.llm_model = llm_model                  # Qwen2-7B-Instruct
        self.model_name_or_path = f'/home/LLMs/Qwen/{self.llm_model}'
        self.template = 'qwen'
        self.device = 4

        self.dataset_path = os.path.join(project_root,'dataset',self.name)       # dataset/Visual_OCR
        self.trainset_path = os.path.join(self.dataset_path,'trainset.json')
        self.validset_path = os.path.join(self.dataset_path,'validset.json')
        self.testset_path = os.path.join(self.dataset_path,'testset.json')

        self.llama_factory_path = os.path.join(project_root,'gectoolkit/llm/LLaMA-Factory')           # LLaMA-Factory
        self.llm_data_path = os.path.join(self.llama_factory_path,'data')                             # LLaMA-Factory/data
        self.llm_dataset_info_path = os.path.join(self.llm_data_path,'dataset_info.json')             # LLaMA-Factory/dataset_info.json
        self.llm_dataset_info = read_json(self.llm_dataset_info_path)

        self.train_lora_path = 'examples/train_lora'                                                  # examples/train_lora
        self.qiwen_lora_predict_path = os.path.join(self.train_lora_path,'qiwen_lora_predict.yaml')   # examples/train_lora/qiwen_lora_predict.yaml
        self.qiwen_lora_sft_path = os.path.join(self.train_lora_path,'qiwen_lora_sft.yaml')           # examples/train_lora/qiwen_lora_sft.yaml

        self.log_path = os.path.join(project_root,'log')                                              # log
        self.log_file = os.path.join(self.log_path, f'{self.llm_model}-{self.name}.log')              # log/Qwen2-7B-Instruct-Visual_OCR.log

        self.checkpoint = os.path.join(project_root,'checkpoint')                                     # checkpoint
        self.checkpoint_path = os.path.join(self.checkpoint,f'{self.llm_model}-{self.name}')          # checkpoint/Qwen2-7B-Instruct-Visual_OCR
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.finetune_sft_dir = f'saves/{self.llm_model}-{self.name}/lora/sft'
        self.finetune_predict_output_dir = f'saves/{self.llm_model}-{self.name}/lora/predict'

        self.setup_logging()                    # make log file
        self.llm_preprocess()                   # prepare data
        self.finetune_llama_factory()            # finetune llm model
        self.finetune_predict_llama_factory()    # finetune predict llm model


    def setup_logging(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                pass
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logging.Formatter.converter = time.gmtime
        logging.info('------------------------- Logging started ---------------------')
        logging.info('Method: %s', self.method)
        logging.info('Name: %s', self.name)
        logging.info('LLM Model: %s', self.llm_model)
        logging.info('Template: %s', self.template)



    def construct_data(self,input_path,output_path):
        infos = []
        with open(input_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                info = {
                    'input': item['source_text'],
                    'output': item['target_text'],
                    # 'instruction':(
                    #     "你是一位经验丰富的中文老师，专门纠正学生作文中的语法错误。请根据以下要求进行修正：\n"
                    #     "1. 纠正所有语法和用词错误。\n"
                    #     "2. 尽量保持句子的原始结构和表达方式，不要大幅度改变句子的意思。\n"
                    #     "3. 修正后的句子应该流畅、自然，并且符合标准的中文语法规则。\n"
                    #     "4. 如果有多种可能的修正方式，请选择最常见和最合适的一种。\n"
                    #     "5. 不要添加或删除任何不必要的内容，只专注于纠正错误。\n"
                    #     "6. 修正后的句子应该与原文的主题和风格保持一致。\n"
                    #     "7. 如果原文中有标点符号错误，请一并纠正。\n"
                    #     "8. 如果原文中有错别字，请更正为正确的汉字。\n"
                    #     "9. 如果原文中有冗余或不恰当的表达，请简化或改进。\n"
                    #     "10. 请确保修正后的句子通顺且易于理解。\n"
                    #     "\n"
                    #     "请根据以上要求对以下文本进行修正：\n"
                    #     f"原文: {item['source_text']}\n"
                    #     "修正后的文本: "
                    # )
                    'instruction': (
                        "你是一位经验丰富的中文老师，专门纠正学生作文中的语法错误。请根据以下要求进行修正：\n"
                        "1. 纠正语法和用词错误。\n"
                        "2. 保持句子的原始结构和表达方式，不要大幅度改变句子的意思。\n"
                        "3. 修正后的句子应流畅、自然，符合标准中文语法规则。\n"
                        "\n"
                        "请根据以上要求对以下文本进行修正：\n"
                        f"原文: {item['source_text']}\n"
                        "修正后的文本: "
                    )
                }
                infos.append(info)
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(infos, file, ensure_ascii=False, indent=4)

    def llm_preprocess(self):
        '''
            return:
                | -- LLaMA_Factory/data
                    | -- dataset_info.json
                        {
                            'Visual_OCR_train': {
                                'file_name': ''Visual_OCR/trainset.json
                            }
                        }
                    | -- Visual_OCR
                        | -- trainset.json
        '''
        out_path = os.path.join(self.llm_data_path,self.name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_train = f'{out_path}/trainset.json'
        out_valid = f'{out_path}/validset.json'
        out_test = f'{out_path}/testset.json'

        self.construct_data(self.trainset_path,out_train)
        self.construct_data(self.validset_path,out_valid)
        self.construct_data(self.testset_path,out_test)
        logging.info('Prepare Dataset With Instruction')
        logging.info(f'Prepare {out_train} successfully')
        logging.info(f'Prepare {out_valid} successfully')
        logging.info(f'Prepare {out_test} successfully')

        # register dataset
        new_field = {
            f"{self.name}_train": {
                "file_name": f"{self.name}/trainset.json"
            },
            f"{self.name}_valid": {
                "file_name": f"{self.name}/validset.json"
            },
            f"{self.name}_test": {
                "file_name": f"{self.name}/testset.json"
            }
        }
        self.llm_dataset_info.update(new_field)
        with open(self.llm_dataset_info_path, 'w', encoding='utf-8') as file:
            json.dump(self.llm_dataset_info, file, ensure_ascii=False, indent=4)

        logging.info(f'Register Dataset To {self.llm_dataset_info_path}')
        print(f'Register Dataset To {self.llm_dataset_info_path}')

    def finetune_llama_factory(self):
        total_qiwen_lora_sft_path = os.path.join(self.llama_factory_path,self.qiwen_lora_sft_path)
        with open(total_qiwen_lora_sft_path, 'r') as file:
            config = yaml.safe_load(file)

        config['model_name_or_path'] = self.model_name_or_path
        config['template'] = self.template
        config['dataset'] = f'{self.name}_train'
        config['output_dir'] = self.finetune_sft_dir

        with open(total_qiwen_lora_sft_path, 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)

        os.chdir(self.llama_factory_path)
        logging.info(f'Turn to {self.llama_factory_path}')

        activate_env_command = "source activate llama_factory"
        logging.info('Activate Env Command')

        print('----------- Finetune -----------')
        print(f'Parameters updated to {self.qiwen_lora_sft_path}')
        print('Please run above commands:\n')
        print('source activate llama_factory')
        print(f'cd {self.llama_factory_path}')
        print(f'CUDA_VISIBLE_DEVICES={self.device} llamafactory-cli train {self.qiwen_lora_sft_path}\n')


    def finetune_predict_llama_factory(self):
        total_qiwen_lora_predict_path = os.path.join(self.llama_factory_path,self.qiwen_lora_predict_path)
        with open(self.qiwen_lora_predict_path, 'r') as file:
            config = yaml.safe_load(file)

        config['model_name_or_path'] = self.model_name_or_path
        config['adapter_name_or_path'] = self.finetune_sft_dir
        config['template'] = self.template
        config['eval_dataset'] = f'{self.name}_test'
        config['output_dir'] = self.finetune_predict_output_dir

        with open(total_qiwen_lora_predict_path, 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)

        print('------ Finetune Predict ---------')
        print(f'Parameters updated at {self.qiwen_lora_predict_path}')
        print('Please run above commands:\n')
        print('source activate llama_factory')
        print(f'cd {self.llama_factory_path}')
        print(f'CUDA_VISIBLE_DEVICES={self.device} llamafactory-cli train {self.qiwen_lora_predict_path}\n')


if __name__ == '__main__':

    ''' only inference'''
    testset_path = '../gectoolkit/llm/LLaMA-Factory/data/Visual_text/testset.json'

    size = 'Qwen2-7B-no-sft'
    model = f'{size}-Instruct_Visual_text'

    checkpoint_path = f'../checkpoint/{model}'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    predict_path = os.path.join(checkpoint_path, 'predicts.json')

    client = OpenAI(api_key="0", base_url="http://0.0.0.0:8000/v1")
    with open(testset_path,'r',encoding='utf-8') as file:
        data = json.load(file)

    results = []
    for item in data:
        source = item['input']
        target = item['output']

        construct = item['instruction']
        messages = [{"role": "user", "content": construct}]
        result = client.chat.completions.create(messages=messages,model=size)
        print(result.choices[0].message)

        info = {
            'source': source,
            'target': target,
            'predict': result.choices[0].message.content
        }
        results.append(info)

    with open(predict_path,'w',encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    print(f'{predict_path} successfully created')
