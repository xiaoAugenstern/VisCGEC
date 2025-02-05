from ultralytics import YOLO
import torch
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import os
import json
import sys
import shutil
import cn_clip.clip as clip
import logging
import time
import matplotlib.pyplot as plt
import re


def extract_number(name, pattern):
    match = re.search(pattern, name)
    if match:
        return int(match.group(1))
    return -1

def replace_punctuation(text):
    # 定义英文标点和对应的中文标点
    punctuation_mapping = {
        ',': '，',
        '.': '。',
        '?': '？',
        '!': '！',
        ':':'：',
        ';':'；'
    }

    # 遍历映射表，替换文本中的标点符号
    for en, zh in punctuation_mapping.items():
        text = text.replace(en, zh)
    return text


def read_json(file_path):
    """Read a JSON file and return its content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


class YoloClipProcessor:
    def __init__(self):
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)  # method
        project_root = os.path.dirname(current_dir)

        self.device = "cuda:7" if torch.cuda.is_available() else "cpu"
        self.dataset_path = os.path.join(project_root, 'dataset')                   # dataset
        self.checkpoint_path = os.path.join(project_root, 'checkpoint')             # checkpoint
        self.log_path = os.path.join(project_root,'log')                            # log
        self.log_file = os.path.join(self.log_path, 'yolo_clip_detect.log')         # log/yolo_clip_detect.log
        self.yolo_clip_path = os.path.join(self.dataset_path,'Visual_YOLO_CLIP')    # dataset/Visual_YOLO_CLIP
        self.visual_image_path = os.path.join(self.dataset_path,'Visual_image')     # dataset/Visual_image

        self.trainset_path = os.path.join(self.yolo_clip_path,'trainset.json')      # Visual_YOLO_CLIP/trainset.json
        self.testset_path = os.path.join(self.yolo_clip_path,'testset.json')        # Visual_YOLO_CLIP/testset.json
        self.validset_path = os.path.join(self.yolo_clip_path,'validset.json')      # Visual_YOLO_CLIP/validset.json

        self.train_data = read_json(self.trainset_path)
        self.valid_data = read_json(self.validset_path)
        self.test_data = read_json(self.testset_path)

        self.detect_location = os.path.join(self.yolo_clip_path,'detect_location')  # Visual_YOLO_CLIP/detect_location
        self.split_image = os.path.join(self.yolo_clip_path,'split_image')          # Visual_YOLO_CLIP/split_image
        self.error = os.path.join(self.detect_location,'error')                     # Visual_YOLO_CLIP/detect_location/error

        self.dict_path = os.path.join(self.checkpoint_path,'cn_clip_checkpoint','dict.txt')
        with open(self.dict_path, 'r', encoding='utf-8') as dict_file:
            self.dict_list = dict_file.read().split('\n')

        self.yolo_model_path = os.path.join(self.checkpoint_path,'yolov8_checkpoint','yolov8_best.pt')
        self.clip_model_path = os.path.join(self.checkpoint_path,'cn_clip_checkpoint','clip_epoch_latest.pt')
        self.all_path = os.path.join(self.dataset_path,'1001','all.json')

        self.yolo_model = YOLO(self.yolo_model_path)                          # load yolov8 checkpoint
        self.clip_model, self.clip_preprocess = self.load_clip_checkpoint()   # load clip checkpoint

        self.only_yolo_dectect = False
        self.only_clip_predict = False
        self.both_yolo_clip = False

        # self.setup_logging()
        # if self.only_yolo_dectect:
        #     self.yolo_detect_location()
        # elif self.only_clip_predict:
        #     self.clip_predict_folder_text()
        # elif self.both_yolo_clip:
        #     self.yolo_detect_location()
        #     self.clip_predict_folder_text()
        #
        # self.update_target_sgt()
        # self.check_yolo_detect()

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

    def load_clip_checkpoint(self):
        model, preprocess = clip.load_from_name("ViT-B-16",
                                                device=self.device,
                                                download_root=os.path.join(self.checkpoint_path,'cn_clip_checkpoint'))
        checkpoint = torch.load(self.clip_model_path, map_location="cpu")
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
        model.load_state_dict(sd)
        model.to(self.device)
        logging.info(f'------- Loading CLIP model from {self.clip_model_path} ')
        logging.info(f'------- Loading candidate text list from {self.dict_path}')
        return model, preprocess

    def segment_image(self,image_path, location_path, output_dir):
        '''
            input: image_path, location_path, output_dir
            output:
             | -- output_dir
                | -- cropped_0.png
                | -- cropped_1.png
        '''
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        img = cv2.imread(image_path)
        # sort according to x1
        with open(location_path, 'r', encoding='utf-8') as f:
            locations = json.load(f)
        locations.sort(key=lambda loc: loc['x1'])
        for i, location in enumerate(locations):
            x1 = location['x1']
            y1 = location['y1']
            x2 = location['x2']
            y2 = location['y2']
            width = x2 - x1
            height = y2 - y1
            padding_ratio = 0.1
            padding = int(max(width, height) * padding_ratio)
            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(img.shape[1], x2 + padding)
            y2_padded = min(img.shape[0], y2 + padding)
            cropped_img = img[y1_padded:y2_padded, x1_padded:x2_padded]
            output_path = os.path.join(output_dir, f'cropped_{i}.png')
            cv2.imwrite(output_path, cropped_img)
        print(f'---- {image_path} segment succesfully!!!')

    def yolo_detect_and_draw(self, img_path,output_path=None):
        '''
             only detect and draw bounding box
        '''
        results = self.yolo_model([img_path, ])    # yolov8 detect
        img = cv2.imread(img_path)
        boxes = results[0].boxes
        colors = {0: (0, 255, 0), 1: (0, 0, 255)}  # 颜色定义
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls)
            color = colors.get(class_id, (0, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if output_path is not None:
            # Save the image to the specified path
            cv2.imwrite(output_path, img)
        else:
            # Show the image using Matplotlib
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()


    def yolo_detect_segment(self,img_path, out_img_path, out_location_path, out_segment_path):
        '''
            use yolov8 to detect characters(0) and punc.(1)
            draw results in out_img_path (green:character, red:punc)
            save x1,y1,x2,y2 in out_location_path
            according to location,segment img_path to cropped_i.png
        '''
        results = self.yolo_model([img_path, ])  # yolov8 detect
        img = cv2.imread(img_path)
        colors = {0: (0, 255, 0), 1: (0, 0, 255)}   # class-0-ch-green,class-1-p-red
        for result in results:
            boxes = result.boxes
            boxes_info = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)  # 0:ch  1:p
                boxes_info.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'class_id': class_id
                })
                color = colors.get(class_id, (0, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.imwrite(out_img_path, img)
            with open(out_location_path, 'w', encoding='utf-8') as f:
                json.dump(boxes_info, f, ensure_ascii=False, indent=4)
        print(f'---- {img_path} detection succesfully!!!')
        self.segment_image(image_path=img_path, location_path=out_location_path, output_dir=out_segment_path)


    def yolo_detect_location(self):
        '''
            return:
                | -- Visual_YOLO_CLIP
                    | -- dection_location
                        | -- train
                            | -- 512_2_location.json
                            | -- 512_2_yolo.png
                    | -- split_image
                        | -- train
                            | -- 512_2
                                | -- cropped_0.png
        '''

        logging.info(f'====== Yolov8 detection {self.visual_image_path} ======')
        for subdir in tqdm(os.listdir(self.visual_image_path)):
            subdir_path = os.path.join(self.visual_image_path, subdir)  # Visual_image/train
            dataset_type = subdir                                       # train,valid,test

            if not os.path.isdir(subdir_path):
                logging.warning(f'{subdir_path} is not a directory. Skipping.')
                continue

            for img in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, img)  # Visual_image/train/512_3.png
                id = img.split('.')[0]                     # 512_3
                out_img_path = os.path.join(self.detect_location, dataset_type,f'{id}_yolo.png')            # Visual_YOLO_CLIP/detect_location/train/512_3_yolo.png
                out_location_path = os.path.join(self.detect_location, dataset_type,f'{id}_location.json')  # Visual_YOLO_CLIP/detect_location/train/512_3_location.json
                out_segment_path = os.path.join(self.split_image, dataset_type,id)                          # Visual_YOLO_CLIP/split_image/train/512_3

                if not os.path.exists(out_segment_path):
                    os.makedirs(out_segment_path)

                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        self.yolo_detect_segment(img_path=img_path,
                                            out_img_path=out_img_path,
                                            out_location_path=out_location_path,
                                            out_segment_path=out_segment_path)
                        logging.info(f'Successfully yolo detect {img_path}')
                    except Exception as e:
                        logging.error(f'Error yolo detect {img_path}: {e}')
                else:
                    logging.warning(f'Unsupported image format: {img_path}')

    def clip_predict_one_image(self, img_path):
        '''
            input: img_path
            return : clip predict top1_label and top1_probs
        '''
        image = self.clip_preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
        text = clip.tokenize(self.dict_list).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            logits_per_image, logits_per_text = self.clip_model.get_similarity(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # top 10 label and probs
        top_indices = probs[0].argsort()[-10:][::-1]
        top_labels = [self.dict_list[i] for i in top_indices]
        top_probs = probs[0][top_indices]

        # top 1 label
        top1_label = top_labels[0]
        top1_probs = top_probs[0]
        return top1_label, top1_probs

    def clip_predict_folder(self,img_id, folder_path):
        '''
            input: img_id , folder_path
            return:
            {
                'img_id': img_id
                'source_text': source_text
            }
        '''
        img_info = {}
        img_info['img_id'] = img_id
        img_info['source_text'] = ''

        for img in os.listdir(folder_path):
            if img.endswith(".jpg") or img.endswith(".png") or img.endswith(".jpeg") or img.endswith(".JPG"):
                img_path = os.path.join(folder_path, img)
                top1_label, top1_probs = self.clip_predict_one_image(img_path=img_path)
                img_info['source_text'] += top1_label

        logging.info(f'---- {img_path} prediction {img_info["source_text"]}')
        print(f'\n{img_id} clip prediction: {img_info["source_text"]}')
        return img_info


    def clip_predict_folder_text(self):
        '''
            clip predict ['source_text'] writing to trainset.json, validset.json, testset.json
        '''
        for subdir in os.listdir(self.split_image):
            images_info = []
            dataset_type = subdir  # train/valid/test
            dataset_path = os.path.join(self.split_image, dataset_type)                # split_image/train

            logging.info(f'------------------------ Clip Predict {dataset_type} ----------------------------')
            for index, segment_folder in enumerate(tqdm(os.listdir(dataset_path))):
                try:
                    segment_folder_path = os.path.join(dataset_path, segment_folder)   # split_image/train/512_5
                    img_info = self.clip_predict_folder(img_id=segment_folder,folder_path=segment_folder_path)
                    images_info.append(img_info)
                    logging.info(f'Successfully clip predict {index} --- {segment_folder}')
                except Exception as e:
                    logging.error(f'Error clip predict {dataset_type} --- {segment_folder}: {e}')

            dataset_file = None
            if dataset_type == 'train':
                dataset_file = self.trainset_path
            elif dataset_type == 'test':
                dataset_file = self.testset_path
            elif dataset_type == 'valid':
                dataset_file = self.validset_path

            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(images_info, f, ensure_ascii=False, indent=4)
            logging.info(f'Successfully clip predict {len(images_info)} images in {dataset_file}')


    def split_image_chunks(self, image_path, chunk_width):
        """
            input: image_path,chunk_width
            output:
                | -- detect_location/error/{img_id}
                    | -- chunk_0.png
                    | -- chunk_1.png
        """
        img_id = image_path.split('/')[-1].split('.')[0]
        output_dir = os.path.join(self.error, img_id)
        os.makedirs(output_dir, exist_ok=True)

        img = cv2.imread(image_path)
        height, width, _ = img.shape

        # 计算裁剪的块数
        num_chunks = width // chunk_width + (1 if width % chunk_width != 0 else 0)
        chunks = []
        for i in range(num_chunks):
            start_x = i * chunk_width
            end_x = min((i + 1) * chunk_width, width)
            chunk = img[:, start_x:end_x, :]
            output_path = os.path.join(output_dir, f"chunk_{i}.png")
            cv2.imwrite(output_path, chunk)
            # print(f"Saved {output_path}")
            chunks.append((output_path,i))
        return chunks

    def detect_on_chunks(self, image_path, chunk_width):
        """
            input: image_path,chunk_width
            return:
                | -- detect_location/error/{img_id}
                    | -- detect_chunk_0.png
                    | -- detect_chunk_0.json
                    | -- segment_chunk_0
                        | -- cropped_0.png
        """
        img_id = image_path.split('/')[-1].split('.')[0]   # 840_8
        output_dir = os.path.join(self.error, img_id)      # detection_location/error/840_8
        text_path = os.path.join(output_dir, 'text.json')
        # yolo detect
        chunks= self.split_image_chunks(image_path, chunk_width)
        for img_path, index in chunks:
            print(img_path)
            out_img_path = os.path.join(output_dir, f"detect_chunk_{index}.png")
            out_location_path = os.path.join(output_dir, f"detect_chunk_{index}.json")
            out_segment_path = os.path.join(output_dir, f'segment_chunk_{index}')
            self.yolo_detect_segment(img_path=img_path,out_img_path=out_img_path,out_location_path=out_location_path,out_segment_path=out_segment_path)

        # clip predict
        segment_dirs = sorted(
            [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))],
            key=lambda d: extract_number(d, r'segment_chunk_(\d+)')
        )
        segment_texts = []
        for segment_dir in segment_dirs:
            segment_dir_path = os.path.join(output_dir, segment_dir)  # segment_chunk_0
            image_files = sorted(
                [f for f in os.listdir(segment_dir_path) if f.endswith('.png')],
                key=lambda f: extract_number(f, r'cropped_(\d+).png')
            )
            text = ''
            for img_file in image_files:
                img_path = os.path.join(segment_dir_path, img_file)   # segment_chunk_0/cropped_0.png
                top1_label, top1_probs = self.clip_predict_one_image(img_path)
                text = text + top1_label
            segment_texts.append(text)

        source_text = replace_punctuation(''.join(segment_texts))
        print(f'{img_id} source_text:', )
        print(source_text)
        with open(text_path, 'w', encoding='utf-8') as f:
            json.dump(source_text, f, ensure_ascii=False, indent=4)


    def check_yolo_detect(self):
        '''
            return:
                error_train_id,error_test_id,error_valid_id
                error_id source_text
        '''
        train_data = read_json(self.trainset_path)
        valid_data = read_json(self.validset_path)
        test_data = read_json(self.testset_path)
        error_train_id = []
        error_valid_id = []
        error_test_id = []
        for item in train_data:
            if item['source_text'] == '':
                error_train_id.append(item['img_id'])
        for item in valid_data:
            if item['source_text'] == '':
                error_valid_id.append(item['img_id'])
        for item in test_data:
            if item['source_text'] == '':
                error_test_id.append(item['img_id'])
        logging.info('this img_id source_text is none,please check')
        logging.info(f'error_train_id:{error_train_id}')
        logging.info(f'error_valid_id:{error_valid_id}')
        logging.info(f'error_test_id:{error_test_id}')

        # for img_id in error_train_id:
        #     img_path = os.path.join(self.visual_image_path, 'train',f'{img_id}.png')
        #     self.detect_on_chunks(image_path=img_path, chunk_width=640)
        # for img_id in error_valid_id:
        #     img_path = os.path.join(self.visual_image_path, 'valid',f'{img_id}.png')
        #     self.detect_on_chunks(image_path=img_path, chunk_width=640)
        # for img_id in error_test_id:
        #     img_path = os.path.join(self.visual_image_path, 'test',f'{img_id}.png')
        #     self.detect_on_chunks(image_path=img_path, chunk_width=640)


    def update_yolo_clip_dataset_with_target(self,input_path, target_path):
        '''
            input: input_path,target_path,output_path
            return:
                input_path add ['id'] ['target_text']  ['source_ground_truth']
        '''
        input_file = read_json(input_path)   # Visual_YOLO_CLIP/trainset.json
        target = read_json(target_path)      # dataset/1001/all.json
        new_infos = []
        for item in input_file:
            img_id = item['img_id']
            source_text = item['source_text']
            for target_item in target:
                if target_item['img_id'] == img_id:
                    new_info = {
                        'id': target_item['id'],
                        "img_id": img_id,
                        'source_text': replace_punctuation(source_text),
                        "target_text": replace_punctuation(target_item['target']),
                        'source_ground_truth': replace_punctuation(target_item['source_ground_truth']),
                    }
                    new_infos.append(new_info)
        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump(new_infos, f, ensure_ascii=False, indent=4)

    def update_target_sgt(self):
        paths = [
            (self.trainset_path, self.all_path),
            (self.validset_path, self.all_path),
            (self.testset_path, self.all_path)
        ]

        for input_path, target_path in paths:
            try:
                self.update_yolo_clip_dataset_with_target(input_path, target_path)
                print(f'{input_path} successfully updated [target_text] [source_ground_truth]')
            except Exception as e:
                print(f'Failed to update {input_path}: {e}')

    def post_clean(self):
        def clean_text(text):
            punctuation = ['：', '，', '、',' ','。']
            while text and text[0] in punctuation:
                text = text[1:]

            if text and len(text) > 1 and text[-1] in punctuation and text[-2] == text[-1]:
                text = text[:-1]

            text = text.replace('，，','，')
            text = text.replace('。。','。')
            text = text.replace('、、','、')
            return text

        def process_data(data):
            for item in data:
                source_text = item['source_text']
                target_text = item['target_text']
                img_id = item['img_id']
                cleaned_text = clean_text(source_text)
                item['source_text'] = cleaned_text
                if source_text != cleaned_text:
                    print(f'img_id: {img_id}')
                    print(f'Original source_text: {source_text}')
                    print(f'Cleaned  source_text: {cleaned_text}')
                    print(f'Original target_text: {target_text}')

        process_data(self.train_data)
        process_data(self.valid_data)
        process_data(self.test_data)

        with open(self.trainset_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_data, f, ensure_ascii=False, indent=4)
        with open(self.validset_path, 'w', encoding='utf-8') as f:
            json.dump(self.valid_data, f, ensure_ascii=False, indent=4)
        with open(self.testset_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_data, f, ensure_ascii=False, indent=4)
        logging.info('Post clean trainset.json')
        logging.info('Post clean validset.json')
        logging.info('Post clean testset.json')


if __name__ == '__main__':
    dataset_path = '../dataset'
    log_path = '../log'
    checkpoint_path = '../checkpoint'
    processor = YoloClipProcessor()

    ''' detect '''
    img_path = '../img.png'
    output_path = '../img_yolo.png'
    processor.yolo_detect_and_draw(img_path,output_path=output_path)

    ''' image width too long'''
    # img_path = '../dataset/Visual_image/train/1126_5.png'
    # processor.detect_on_chunks(image_path=img_path,chunk_width=640)

    '''check'''
    # processor.check_yolo_detect()

    ''' clean punctuation'''
    # processor.post_clean()