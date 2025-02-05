import json
from transformers import pipeline, AutoProcessor
from PIL import Image
import requests
import os
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def make_dataset(input_path,type,output_path):
    with open(input_path,'r',encoding='utf-8') as f:
        data = json.load(f)

    result = []
    for item in data:
        source_text = item['source_text']
        target_text = item['target_text']
        img_id = item['img_id']
        images = "/home/xiaoman/project/gec/HandwrittenGEC/dataset/Visual_image/" + type + "/" + img_id + ".png"
        info = {
            'instruction': (
                "你是一位经验丰富的中文老师，专门纠正学生作文中的语法错误。输入包括一张包含语法错误的图片<image>以及该图片中提取的文本内容。请根据以下要求对文本进行修正：\n"
                "1. 根据提供的纠正语法和用词错误。\n"
                "2. 保持句子的原始结构和表达方式，不要大幅度改变句子的意思。\n"
                "3. 修正后的句子应流畅、自然，符合标准中文语法规则。\n"
                "\n"
                "请根据以上要求对以下文本进行修正：\n"
                f"原文: {source_text}\n"
                "修正后的文本: "
            ),
            "input": source_text,
            "output": target_text,
            "images": [
                images
            ]
        }
        result.append(info)

    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(result,f,indent=4,ensure_ascii=False)
        print(f'make dataset {output_path} successfully!')



def check(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 获取所有图片路径
    image_paths = [item['images'][0] for item in data]

    # 检查图片是否存在
    def check_images_exist(image_paths):
        results = {}
        for path in image_paths:
            results[path] = os.path.exists(path)
        return results

    # 执行检查并打印结果
    results = check_images_exist(image_paths)
    for path, exists in results.items():
        if exists == False:
            print(f"图片 {path} 不存在")


def llava_predict(json_file_path, output_file_path):
    # 模型路径
    model_id = "/home/LLMs/llava/llava-1.5-7b-hf"
    # 初始化模型和处理器
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)
    processor = AutoProcessor.from_pretrained(model_id)

    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    results = []

    for index,item in enumerate(data):
        instruction = item['instruction']
        input_text = item['input']
        output_text = item['output']
        image_path = item['images'][0]

        # 构建对话
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image"},
                ],
            },
        ]

        # 生成提示
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        # 加载本地图像
        raw_image = Image.open(image_path)

        # 处理图像和文本
        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

        # 生成输出
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

        # 解码输出
        predicted_text = processor.decode(output[0][2:], skip_special_tokens=True)
        print('index:',index)
        print('source:', input_text)
        print('target:', output_text)
        print('predict:',predicted_text)
        print()
        
        # 格式化结果
        result = {
            "source": input_text,
            "target": output_text,
            "predict": predicted_text
        }

        results.append(result)

    # 将结果写入输出文件
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)
    print('successfully write to {output_file_path}'.format(output_file_path=output_file_path))



def use_transformers():
    model_id = "/home/LLMs/llava/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)
    processor = AutoProcessor.from_pretrained(model_id)
    conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "你是一位经验丰富的中文老师，专门纠正学生作文中的语法错误。输入包括一张包含语法错误的图片以及该图片中提取的文本内容。请根据以下要求对文本进行修正：\n1. 根据提供的纠正语法和用词错误。\n2. 保持句子的原始结构和表达方式，不要大幅度改变句子的意思。\n3. 修正后的句子应流畅、自然，符合标准中文语法规则。\n\n请根据以上要求对以下文本进行修正：\n原文: 这些的店比较有自由感，可以随便穿穿。\n修正后的文本: "},
                        {"type": "image"},
                    ],
                },
            ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # 本地图像路径
    image_path = "/home/xiaoman/project/gec/HandwrittenGEC/dataset/Visual_image/test/813_4.png"

    # 加载本地图像
    raw_image = Image.open(image_path)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    print(processor.decode(output[0][2:], skip_special_tokens=True))


if __name__ == '__main__':
    visual_ocr_train_path = '../dataset/Visual_OCR/trainset.json'
    visual_ocr_test_path = '../dataset/Visual_OCR/testset.json'
    visual_ocr_valid_path = '../dataset/Visual_OCR/validset.json'

    output_train_path = '../gectoolkit/llm/LLaMA-Factory/data/Visual_OCR/trainset_image.json'
    output_valid_path = '../gectoolkit/llm/LLaMA-Factory/data/Visual_OCR/validset_image.json'
    output_test_path = '../gectoolkit/llm/LLaMA-Factory/data/Visual_OCR/testset_image.json'

    '''制作数据集'''
    # make_dataset(visual_ocr_train_path,'train',output_train_path)
    # make_dataset(visual_ocr_valid_path,'valid',output_valid_path)
    # make_dataset(visual_ocr_test_path,'test',output_test_path)
    #
    # check(output_train_path)
    # check(output_valid_path)
    # check(output_test_path)

    '''pipeline'''
    # use_pipeline()

    use_transformers()

    '''' inference '''
    # llava_predict(output_test_path, '../checkpoint/LLava1_5-no-sft-Visual_OCR/llava_predict.json')