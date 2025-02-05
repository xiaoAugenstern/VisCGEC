import argparse
from PIL import Image
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))
from gectoolkit.quick_start import run_toolkit
from method.gpt4 import run_gpt4
from method.yolov8_clip import YoloClipProcessor
from method.llm import LLM
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-d', type=str, default='Visual', help='name of datasets')
    parser.add_argument('--method', choices=['OCR', 'YOLO_CLIP', 'GPT4v','text'], default='YOLO_CLIP', required=False, help='Choose the method: ocr, yolo_clip, or gpt4v.')
    parser.add_argument('--correction-model', choices=['SynGEC', 'GECToR', 'qianwen'], default='qianwen', required=False, help='Choose the correction model.')
    parser.add_argument('--augment', type=str, default='none', choices=['none', 'noise', 'translation'],help='use data augmentation or not')

    # 解析命令行参数
    args = parser.parse_args()

    if args.method == 'GPT4v':
        args.correction_model = 'None'

    # 打印解析后的参数
    print('\n----------------------------------------------------')
    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.method}")
    print(f"Correction Model: {args.correction_model}")
    print('----------------------------------------------------')


    qiwen_model = 'Qwen2-7B-Instruct'
    if args.method == 'OCR' or args.method == 'text':
        if args.correction_model == 'SynGEC' or args.correction_model == 'GECToR':
            run_toolkit(method_name=args.method, model_name=args.correction_model, dataset_name=args.dataset, augment_method=args.augment)
        elif args.correction_model == 'qianwen':
            processor = LLM(method=args.method, llm_model=qiwen_model)

    elif args.method == 'YOLO_CLIP':
        # detect = False
        # if not detect:
        #     processor = YoloClipProcessor()
        if args.correction_model == 'SynGEC' or args.correction_model == 'GECToR':
            run_toolkit(method_name=args.method, model_name=args.correction_model, dataset_name=args.dataset,augment_method=args.augment)
        elif args.correction_model == 'qianwen':
            processor = LLM(method=args.method, llm_model=qiwen_model)

    elif args.method == 'GPT4v':
        result_path = 'checkpoint/GPT4-Visual_image/predicts.json'
        visual_image_test_path = 'dataset/Visual_image/test'
        gpt4_log_file_path = 'log/Visual_GPT4v.log'
        run_gpt4(visual_image_test_path,result_path,log_file=gpt4_log_file_path)
    else:
        print('Please choose the method:ocr, yolo_clip, text, or gpt4v.')
        run_toolkit(method_name=args.method, model_name=args.correction_model, dataset_name=args.dataset,augment_method=args.augment)


