import json
import os
from openai import OpenAI
from tqdm import tqdm
import base64
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# os.environ["http_proxy"] = "http://localhost:7890"
# os.environ["https_proxy"] = "http://localhost:7890"

client = OpenAI(
    api_key = 'sk-proj-8PnZY0t8qmNR2PjYOT0TT3BlbkFJi8OFVzUV3pEa2kX6LCC7',
)

# 配置日志记录
def setup_logging(log_file):
    # 确保日志文件存在
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            pass  # 创建一个空的日志文件

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 配置日志记录器
    logging.basicConfig(
        filename=log_file,  # 日志文件路径
        filemode='a',       # 追加模式
        level=logging.INFO, # 设置日志级别
        format='%(asctime)s - %(levelname)s - %(message)s', # 日志格式
        datefmt='%Y-%m-%d %H:%M:%S'  # 时间格式
    )

def correct_text_from_image(prompt, img_url=None, img_path=None):
    if img_path:
        # 将图像文件读取为 Base64 字符串
        with open(img_path, 'rb') as img_file:
            img_b64_str = base64.b64encode(img_file.read()).decode('utf-8')
        img_type = 'image/png'  # 根据实际图像类型调整

        # 构建请求
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
                        },
                    ],
                }
            ],
        )
    elif img_url:
        # 使用图像 URL 进行请求
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": img_url},
                        },
                    ],
                }
            ],
        )
    else:
        raise ValueError("Either img_url or img_path must be provided.")

    return response

def run_gpt4(img_root_path,output_path,log_file):
    setup_logging(log_file)
    prompt = (
        "你是一位经验丰富的中文老师，专门纠正学生作文中的语法错误。"
        "请查看以下包含语法错误的小照片，完成以下任务：\n"
        "1. 识别并提取照片中的文本内容。\n"
        "2. 纠正所有语法和用词错误，尽量保持句子的原始结构和表达方式。\n"
        "3. 列出具体的修改操作，例如：将 '错误的词语' 修改为 '正确的词语'。\n"
        "请按照以下格式输出结果：\n"
        "1. 识别的结果: [识别的文本内容]\n"
        "2. 纠正后的正确结果: [纠正后的文本内容]\n"
        "3. 修改操作: [具体的修改操作列表]"
    )
    results = []
    print('************************* gpt4 process ************************* ')
    logging.info('Starting gpt4 process...')

    for img_path in tqdm(os.listdir(img_root_path)):
        info = {}
        id = img_path.split('.')[0]
        img_path = os.path.join(img_root_path, img_path)

        if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                response = correct_text_from_image(prompt, img_path, img_path)
                corrected_output = response.choices[0].message.content

                info['img_id'] = id
                info['corrected_output'] = corrected_output
                print(f'{id} corrected_output:', corrected_output)
                logging.info(f'------------{id}------------')
                logging.info(f'{corrected_output}')
                results.append(info)
            except Exception as e:
                logging.info('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                logging.error(f'Error processing image {img_path}: {e}')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logging.info(f'gpt4v results saved to {output_path} successfully.')
    print(f'************************* gpt4v results saved to {output_path} successfully. *************************')


def replace_punctuation(text):
    # 定义英文标点和对应的中文标点
    punctuation_mapping = {
        ',': '，',
        '.': '。',
        '?': '？',
        '!': '！'
    }

    # 遍历映射表，替换文本中的标点符号
    for en, zh in punctuation_mapping.items():
        text = text.replace(en, zh)
    return text

def process_predicts(input_path,output_path):
    # 读取输入文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    error_id = []

    for item in data:
        img_id = item['img_id']
        corrected_output = item['corrected_output']
        if '抱歉' in corrected_output:
            print(f'{img_id} 存在抱歉')
            error_id.append(img_id)
        else:
            # 分割 corrected_output 为三个部分
            parts = corrected_output.split('\n\n')

            if len(parts) != 3:
                parts = corrected_output.split('\n')
                print(f'{img_id}的parts为：{parts}')

            # 提取识别结果
            detection_result = parts[0].replace('1. 识别的结果: ', '')

            # 提取纠正后的结果
            prediction = parts[1].replace('2. 纠正后的正确结果: ', '')

            # 提取修改操作
            edits_str = parts[2].replace('3. 修改操作:', '').strip()
            # 将修改操作按行分割，并去除每行的前导空格
            edits = [edit.strip() for edit in edits_str.split('\n') if edit.strip()]

            # 创建新的条目并添加到处理后的数据列表中
            processed_item = {
                'img_id': item['img_id'],
                'detection_result': replace_punctuation(detection_result),
                'prediction': replace_punctuation(prediction),
                'edits': edits
            }
            processed_data.append(processed_item)

        # 写入输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)
        print(f'{output_path} successfully processed.')

    print('erro_id',error_id)

def qiwen_model():
    torch.manual_seed(1234)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    prompt = (
        "你是一位经验丰富的中文老师，专门纠正学生作文中的语法错误。"
        "请查看以下包含语法错误的小照片，完成以下任务：\n"
        "1. 识别并提取照片中的文本内容。\n"
        "2. 纠正所有语法和用词错误，尽量保持句子的原始结构和表达方式。\n"
        "3. 列出具体的修改操作，例如：将 '错误的词语' 修改为 '正确的词语'。\n"
        "请按照以下格式输出结果：\n"
        "1. 识别的结果: [识别的文本内容]\n"
        "2. 纠正后的正确结果: [纠正后的文本内容]\n"
        "3. 修改操作: [具体的修改操作列表]"
    )
    testset_path = '../dataset/Visual_image/test'
    responses = []
    for item in os.listdir(testset_path):
        img_path = os.path.join(testset_path, item)
        query = tokenizer.from_list_format([
            {'image':img_path},
            {'text':'You are an experienced Chinese teacher who specializes in correcting grammatical errors in students compositions.please recognize and extract the textual content in the photos,correct all grammatical and diction errors and output right sentence.'}
        ])
        response, history = model.chat(tokenizer, query=query, history=None)
        print(response)
        responses.append(response)
    with open('responses.json', 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    # input_path = '../checkpoint/GPT4-Visual_image/predicts.json'
    # output_path = '../checkpoint/GPT4-Visual_image/predicts_process.json'
    # process_predicts(input_path=input_path,output_path=output_path)

    '''gpt4v'''
    # ======================== 使用单一图片，上传gpt4纠错 ==========================
    # 抱歉，我无法处理图像内容。请提供文本，我将乐意帮助你纠正其中的语法错误。
    prompt = (
        "你是一位经验丰富的中文老师，专门纠正学生作文中的语法错误。"
        "请查看以下包含语法错误的小照片，完成以下任务：\n"
        "1. 识别并提取照片中的文本内容。\n"
        "2. 纠正所有语法和用词错误，尽量保持句子的原始结构和表达方式。\n"
        "3. 列出具体的修改操作，例如：将 '错误的词语' 修改为 '正确的词语'。\n"
        "请按照以下格式输出结果：\n"
        "1. 识别的结果: [识别的文本内容]\n"
        "2. 纠正后的正确结果: [纠正后的文本内容]\n"
        "3. 修改操作: [具体的修改操作列表]"
    )

    img_path = f'/home/xiaoman/project/gec/HandwrittenGEC/dataset/Visual_image/train/2012_2.png'  # 或者 img_url = 'http://example.com/image.png'
    response = correct_text_from_image(prompt, img_path=img_path)
    corrected_output = response.choices[0].message.content
    print(f"{img_path} Corrected Text:\n", corrected_output)

    '''qiwen-vl-chat (multi)'''
    # qiwen_model()
