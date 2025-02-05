from transformers import T5Tokenizer,T5ForConditionalGeneration,T5Config
import numpy as np
import torch
from transformers import MT5Tokenizer
from transformers import MT5ForConditionalGeneration, AutoTokenizer
from torch import nn


class T5(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.device = config["device"]

        # get model path
        self.model_path = config["pretrained_model_path"]

        # self.dataloader.pretrained_tokenzier
        self.tokenizer = dataset

        # load model
        self.model = MT5ForConditionalGeneration.from_pretrained(self.model_path)
        self.pad_token_id = self.tokenizer.pad_token_id  # pad id= 0
        self.test_id = 0

    def forward(self, batch, dataloader):
        """
        parameters in batch.
        batch['source_batch'] is a list with the source text sentences;
        batch['target_batch'] is a list with the target text sentences;
        batch['source_list_batch'] is the indexed source sentences;
        batch['target_list_batch'] is the indexed target sentences;
        """
        language = dataloader.dataset.language_name      # 中文数据集/英文数据集
        source_list_batch = batch["source_list_batch"]
        target_list_batch = batch["target_list_batch"]

        # max len
        source_max_len = np.max([len(sent) for sent in source_list_batch])
        target_max_len = np.max([len(sent) for sent in target_list_batch])
        # print('source_max_len:', source_max_len)
        # print('target_max_len:', target_max_len)

        # chinese source： [['从', '来', '没', '有', '学', '过', '汉', '语', '。'],[],...]
        # english source： [['▁De', 'ar', '▁Mr', '▁...', '▁', ','],....]
        source_data = batch['source_batch']
        target_data = batch['target_batch']


        # chinese source：['从来没有学过汉语。','可是我口语的方面还不好。', ... ]
        # english source：['Firstly I thought','..', ...]
        if language == 'zh':
            source_data = [''.join(words) for words in source_data]
            target_data = [''.join(words) for words in target_data]
        else:
            source_data = self.tokenizer.batch_decode(source_list_batch)
            target_data = self.tokenizer.batch_decode(target_list_batch)

        # source encode
        source_encoding = self.tokenizer.batch_encode_plus(source_data,
                                                           pad_to_max_length=True,
                                                           max_length=source_max_len,
                                                           truncation=True,
                                                           return_tensors="pt")

        # get source_id and attention_mask
        source_ids = source_encoding.input_ids                  # torch.Size([batch_size, source_max_len])
        source_attention_mask = source_encoding.attention_mask  # torch.Size([batch_size, source_max_len])

        if self.device:
            source_ids = source_ids.to(self.device)
            source_attention_mask = source_attention_mask.to(self.device)

        # target encode
        target_encoding = self.tokenizer.batch_encode_plus(target_data,
                                                           pad_to_max_length=True,
                                                           max_length=target_max_len,
                                                           truncation=True,
                                                           return_tensors="pt")

        target_ids = target_encoding.input_ids
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        if self.device:
            target_ids = target_ids.to(self.device)

        # Forward pass
        outputs = self.model(input_ids = source_ids,
                             attention_mask = source_attention_mask,
                             labels = target_ids)
        loss = outputs.loss
        print('loss:',loss)
        logits = outputs.logits  # torch.Size([batch_size, max_length, vocab_size])

        predicted_ids = logits.argmax(dim=-1)

        # decode
        predict_decode = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        # print('target_data   :',target_data[0])
        # print('predict_decode:',predict_decode[0])
        # print('-----loss:', loss)

        loss_dic = {"decode_result": predicted_ids,
                    "loss": loss}

        return loss_dic

    def model_test(self, batch, dataloader):
        with torch.no_grad():
            language = dataloader.dataset.language_name

            source_list_batch = batch["source_list_batch"]
            target_list_batch = batch["target_list_batch"]

            source_max_len = np.max([len(sent) for sent in source_list_batch])
            target_max_len = np.max([len(sent) for sent in target_list_batch])

            # batch，[['图', '书', '馆', '的', '对', '面', '有', '学', '生', '会', '馆', '。']]
            source_data = batch['source_batch']
            target_data = batch['target_batch']

            if language == 'zh':
                source_data = [''.join(words) for words in source_data]
                target_data = [''.join(words) for words in target_data]
            else:
                source_data = self.tokenizer.batch_decode(source_list_batch)
                target_data = self.tokenizer.batch_decode(target_list_batch)

            # source encode
            source_encoding = self.tokenizer.batch_encode_plus(source_data,
                                                               pad_to_max_length=True,
                                                               max_length=source_max_len,
                                                               truncation=True,
                                                               return_tensors="pt")

            source_ids = source_encoding.input_ids                  # torch.Size([1, source_max_len])
            source_attention_mask = source_encoding.attention_mask  # torch.Size([1, source_max_len])

            if self.device:
                source_ids = source_ids.to(self.device)
                # source_attention_mask = source_attention_mask.to(self.device)

            # target encode
            target_encoding = self.tokenizer.batch_encode_plus(target_data,
                                                               pad_to_max_length=True,
                                                               max_length=target_max_len,
                                                               truncation=True,
                                                               return_tensors="pt")
            target_ids = target_encoding.input_ids

            if language == 'zh':
                predicted_ids = self.model.generate(
                    input_ids=source_ids,
                    max_length=target_max_len +10,  # disable sampling to test if batching affects output
                )
            else:
                predicted_ids = self.model.generate(
                    input_ids=source_ids,
                    max_length=target_max_len+2,  #  ValueError: Input length of decoder_input_ids is 1, but `max_length` is set to 1.
                )

            # decode
            target_decode = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)
            predict_decode = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)


            print('\ntarget_decode :',target_decode)
            print('predict_decode:',predict_decode)

            # mt5large存在中英文符号转换问题,将英文符号转成中文
            if language == 'zh':
                # ['图', '书', '馆', '的', '对', '面', '有', '学', '生', '会', '馆', '。']
                predict_char_text = [char for text in predict_decode for char in text]
                # convert into id
                predict_char_ids = self.tokenizer.convert_tokens_to_ids(predict_char_text)

                for i in range(len(predict_char_ids)):
                    if predict_char_ids[i] == self.tokenizer.convert_tokens_to_ids("<unk>"):
                        dataloader.replaced_symbols.append(predict_char_text[i])
                    if predict_char_ids[i] == self.tokenizer.convert_tokens_to_ids(","):
                        predict_char_ids[i] = self.tokenizer.convert_tokens_to_ids("，")
                        dataloader.replaced_symbols.append("，")
                    if predict_char_ids[i] == self.tokenizer.convert_tokens_to_ids("!"):
                        predict_char_ids[i] = self.tokenizer.convert_tokens_to_ids("！")
                        dataloader.replaced_symbols.append("！")
                    if predict_char_ids[i] == self.tokenizer.convert_tokens_to_ids(";"):
                        predict_char_ids[i] = self.tokenizer.convert_tokens_to_ids("；")
                        dataloader.replaced_symbols.append("；")
                    if predict_char_ids[i] == self.tokenizer.convert_tokens_to_ids(":"):
                        predict_char_ids[i] = self.tokenizer.convert_tokens_to_ids("：")
                        dataloader.replaced_symbols.append("：")
                    if predict_char_ids[i] == self.tokenizer.convert_tokens_to_ids("?"):
                        predict_char_ids[i] = self.tokenizer.convert_tokens_to_ids("？")
                        dataloader.replaced_symbols.append("？")
                    result = [predict_char_ids]
            else:
                result = predicted_ids.tolist()

        return result,target_ids
