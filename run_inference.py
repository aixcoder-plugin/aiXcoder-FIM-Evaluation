# -*- coding: utf-8 -*-
import argparse
import os
import json
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from utils import input_wrapper
from tqdm import tqdm

import time

import requests
import datetime
import traceback

from requests.exceptions import RequestException


class Model_Gen:
    def __init__(self, cuda, gen_len, max_len, model_dir, torch_dtype, attn_implementation) -> None:
        self.cuda = cuda
        # self.language = language
        self.model_dir = model_dir
        dict_precisions = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if attn_implementation.lower() == 'true':
            self.model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                              torch_dtype=dict_precisions[torch_dtype],
                                                              attn_implementation="flash_attention_2").to(cuda)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                              torch_dtype=dict_precisions[torch_dtype]).to(cuda)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.max_gen_len = gen_len
        self.half_max_len = (max_len - self.max_gen_len) // 2

    def predict(self, code_string, later_code='\n', language='python'):
        if 'aix' in self.model_dir.lower():
            if language == 'python':
                ext = 'py'
            elif language == 'java':
                ext = 'java'
            elif language == 'javascript':
                ext = 'js'
            elif language == 'cplus':
                ext = 'cpp'
            else:
                ext = 'py'
            inputs_pre = self.tokenizer(code_string, return_tensors="pt", return_token_type_ids=False).to(self.cuda)
            if len(inputs_pre[0]) >= self.half_max_len:
                code_string = self.tokenizer.decode(inputs_pre['input_ids'][0][-self.half_max_len:],
                                                    skip_special_tokens=False)
            inputs_post = self.tokenizer(later_code, return_tensors="pt", return_token_type_ids=False).to(self.cuda)
            if len(inputs_post[0]) >= self.half_max_len:
                later_code = self.tokenizer.decode(inputs_post['input_ids'][0][:self.half_max_len],
                                                   skip_special_tokens=False)
            text = input_wrapper(
                # for FIM style input, code_string stands for prefix context
                code_string=code_string,
                tokenizer=self.tokenizer,
                # for FIM style input, later_code stands for suffix context
                later_code=later_code,
                # file_path should be a path from project to file
                path=f"test.{ext}"  # .cpp/.java/.js/.py
            )

            text = text.to(self.cuda)
            outputs = self.model.generate(**text, max_new_tokens=512, pad_token_id=self.tokenizer.eos_token_id)
            input_len = text['input_ids'].shape[1]
            result = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)
            print('当前为aiXcoder_7b_Service服务')
            if '</s>' in result:
                return result.split('</s>')[0]

            return result
        elif 'deepseek' in self.model_dir.lower():
            text_pre = '<｜fim▁begin｜>' + code_string + '<｜fim▁hole｜>'
            text_post = later_code + '<｜fim▁end｜>'
            inputs_pre = self.tokenizer(text_pre, return_tensors="pt").to(self.cuda)
            inputs_post = self.tokenizer(text_post, return_tensors="pt").to(self.cuda)
            if len(inputs_pre[0]) >= self.half_max_len:
                inputs_pre['input_ids'] = torch.cat(
                    [inputs_pre['input_ids'][:, :2], inputs_pre['input_ids'][:, -self.half_max_len + 2:]], dim=1)
                inputs_pre['attention_mask'] = torch.cat(
                    [inputs_pre['attention_mask'][:, :2], inputs_pre['attention_mask'][:, -self.half_max_len + 2:]],
                    dim=1)
            if len(inputs_post[0]) >= self.half_max_len:
                inputs_pre['input_ids'] = torch.cat(
                    [inputs_pre['input_ids'][:, :], inputs_post['input_ids'][:, 1:self.half_max_len],
                     inputs_post['input_ids'][:, -1:]], dim=1)
                inputs_pre['attention_mask'] = torch.cat(
                    [inputs_pre['attention_mask'][:, :], inputs_post['attention_mask'][:, 1:self.half_max_len],
                     inputs_post['attention_mask'][:, -1:]], dim=1)
            else:
                inputs_pre['input_ids'] = torch.cat([inputs_pre['input_ids'][:, :], inputs_post['input_ids'][:, 1:]],
                                                    dim=1)
                inputs_pre['attention_mask'] = torch.cat(
                    [inputs_pre['attention_mask'][:, :], inputs_post['attention_mask'][:, 1:]], dim=1)
            outputs = self.model.generate(**inputs_pre, max_new_tokens=self.max_gen_len,
                                          pad_token_id=self.tokenizer.eos_token_id)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if '<｜fim▁end｜>' in text:
                return text.split('<｜fim▁end｜>')[-1]
            return text
        elif 'starcoder2' in self.model_dir.lower():
            prefix_id = self.tokenizer("<fim_prefix>", return_tensors="pt").to(self.cuda)
            suffix_id = self.tokenizer("<fim_suffix>", return_tensors="pt").to(self.cuda)
            middle_id = self.tokenizer("<fim_middle>", return_tensors="pt").to(self.cuda)
            eos_id = self.tokenizer("<|endoftext|>", return_tensors="pt").to(self.cuda)
            file_sep = self.tokenizer("<file_sep>", return_tensors="pt").to(self.cuda)
            inputs_pre = self.tokenizer(code_string, return_tensors="pt").to(self.cuda)
            inputs_post = self.tokenizer(later_code, return_tensors="pt").to(self.cuda)
            if len(inputs_pre[0]) >= self.half_max_len:
                inputs_pre['input_ids'] = inputs_pre['input_ids'][:, -self.half_max_len:]
                inputs_pre['attention_mask'] = inputs_pre['attention_mask'][:, -self.half_max_len:]
            if len(inputs_post[0]) >= self.half_max_len:
                inputs_post['input_ids'] = inputs_post['input_ids'][:, :self.half_max_len]
                inputs_post['attention_mask'] = inputs_post['attention_mask'][:, :self.half_max_len]
            if len(inputs_pre[0]) == 0 and len(inputs_post[0]) == 0:
                inputs_pre['input_ids'] = torch.cat(
                    [prefix_id['input_ids'], suffix_id['input_ids'], middle_id['input_ids']], dim=1)
                inputs_pre['attention_mask'] = torch.cat(
                    [prefix_id['attention_mask'], suffix_id['attention_mask'], middle_id['attention_mask']], dim=1)
            elif len(inputs_pre[0]) == 0:
                inputs_pre['input_ids'] = torch.cat(
                    [prefix_id['input_ids'], suffix_id['input_ids'], inputs_post['input_ids'], middle_id['input_ids']],
                    dim=1)
                inputs_pre['attention_mask'] = torch.cat(
                    [prefix_id['attention_mask'], suffix_id['attention_mask'], inputs_post['attention_mask'],
                     middle_id['attention_mask']], dim=1)
            elif len(inputs_post[0]) == 0:
                inputs_pre['input_ids'] = torch.cat(
                    [prefix_id['input_ids'], inputs_pre['input_ids'], suffix_id['input_ids'], middle_id['input_ids']],
                    dim=1)
                inputs_pre['attention_mask'] = torch.cat(
                    [prefix_id['attention_mask'], inputs_pre['attention_mask'], suffix_id['attention_mask'],
                     middle_id['attention_mask']], dim=1)
            else:
                inputs_pre['input_ids'] = torch.cat(
                    [prefix_id['input_ids'], inputs_pre['input_ids'], suffix_id['input_ids'], inputs_post['input_ids'],
                     middle_id['input_ids']], dim=1)
                inputs_pre['attention_mask'] = torch.cat(
                    [prefix_id['attention_mask'], inputs_pre['attention_mask'], suffix_id['attention_mask'],
                     inputs_post['attention_mask'], middle_id['attention_mask']], dim=1)
            input_len = inputs_pre['input_ids'].shape[1]
            outputs = self.model.generate(**inputs_pre, max_new_tokens=512, pad_token_id=self.tokenizer.eos_token_id)
            text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)
            print('当前为StarCoder2_Service服务')
            if '<file_sep>' in text:
                return text.split('<file_sep>')[0]
            return text
        elif 'codellama' in self.model_dir.lower():
            query = code_string + '<FILL_ME>' + later_code
            inputs_text = self.tokenizer(query, return_tensors="pt").to(self.cuda)

            inputs_pre = self.tokenizer(code_string, return_tensors="pt").to(self.cuda)
            inputs_post = self.tokenizer(later_code, return_tensors="pt").to(self.cuda)
            # 截断长度
            truncation_len = self.half_max_len
            # 前文代码长度
            pre_len = len(inputs_pre[0])
            # 后文代码长度
            post_len = len(inputs_post[0])
            if len(inputs_pre[0]) >= truncation_len:
                inputs_text['input_ids'] = torch.cat(
                    [inputs_text['input_ids'][:, :2], inputs_text['input_ids'][:, pre_len - truncation_len:]], dim=1)
                inputs_text['attention_mask'] = torch.cat(
                    [inputs_text['attention_mask'][:, :2], inputs_text['attention_mask'][:, pre_len - truncation_len:]],
                    dim=1)
            if len(inputs_post[0]) >= truncation_len:
                inputs_text['input_ids'] = torch.cat(
                    [inputs_text['input_ids'][:, :-(post_len - truncation_len)], inputs_text['input_ids'][:, -1:]],
                    dim=1)
                inputs_text['attention_mask'] = torch.cat(
                    [inputs_text['attention_mask'][:, :-(post_len - truncation_len)],
                     inputs_text['attention_mask'][:, -1:]], dim=1)

            input_len = inputs_text['input_ids'].shape[1]
            outputs = self.model.generate(**inputs_text, max_new_tokens=512, pad_token_id=self.tokenizer.eos_token_id)
            text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            print('当前为codellama_7b_Service服务')

            return text
        else:
            pass



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"'{path}' has been created")
    else:
        print(f"'{path}' already exists")


def save_result(res_writer, res_json):
    res_writer.write(json.dumps(res_json, ensure_ascii=False) + "\n")
    res_writer.flush()


def main(config):
    service = Model_Gen(cuda=config.device,
                        model_dir=config.model,
                        torch_dtype=config.torch_dtype,
                        attn_implementation=config.attn_implementation,
                        gen_len=config.gen_len, 
                        max_len=config.max_len)

    languages = config.language.split(' ')
    for language in languages:
        print('评测语言：', language)
        if language == 'cpp':
            language == 'cplus'
        source_path = f'datasets/{language}_test_8k_full.jsonl'
        with open(source_path, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()
        start_index = 0
        save_path = f"{config.output_dir}/{config.model.split('/')[-1]}/{language}_{config.model.split('/')[-1]}_8k.jsonl"
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                result_lines = f.readlines()
            start_index = len(result_lines)

        elif os.path.exists(f"{config.output_dir}/{config.model.split('/')[-1]}"):
            pass
        else:
            os.makedirs(f"{config.output_dir}/{config.model.split('/')[-1]}")
        process_bar = tqdm(total=len(raw_lines), desc=f'processin {language}')
        with open(save_path, 'a') as res_writer:
            for idx, line in enumerate(raw_lines):
                try:
                    if idx < start_index:
                        process_bar.update(1)
                        continue
                    line = json.loads(line)
                    res_dict = {}
                    task_id = line['task_id']
                    pre_mask_code = line['pre_mask_code']
                    post_mask_code = line['post_mask_code']
                    prediction = service.predict(pre_mask_code, post_mask_code, language)
                    res_dict["task_id"] = task_id
                    res_dict["prediction"] = prediction
                    res_writer.write(json.dumps(res_dict, ensure_ascii=False) + "\n")
                    res_writer.flush()
                    process_bar.update(1)
                except Exception as e:
                    traceback.print_exc()
                    print(f"Attention !!!!!!  Task {task_id} throw exception !!!!!!")
                    if res_writer:
                        res_writer.close()
                    break
                    # continue
                # finally:
                #     if res_writer:
                #         res_writer.close()
                #     break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 设置模型
    parser.add_argument(
        "--model",
        type=str,
        default="/data2/open_source_aixcoder_7b/aiXcoder-7b-base-weights",
        choices=["aiXcoder/aixcoder-7b-base", "deepseek-ai/deepseek-coder-6.7b-base", "codellama/CodeLlama-7b-hf", "bigcode/starcoder2-7b"],
        help="Type of Model to run"
    )
    # 设置评测的语言
    parser.add_argument(
        "--language",
        type=str,
        default="java cplus javascript",
        choices=["python", "java", "cpp", "javascript"],
        help="Language of Dataset to inference"
    )
    # 设置结果位置
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Path to which the inference results will be cached at",
    )
    # device="cuda"
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device used",
    )
    # torch_dtype=torch.bfloat16
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Precision of Model loaded",
    )
    # attn_implementation="flash_attention_2"
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default='False',
        choices=["True", "False"],
        help="Use FlashAttention or not",
    )
    # gen_len=512
    parser.add_argument(
        "--gen_len",
        type=int,
        default="512",
        help="Length of Generate",
    )
    # max_len=16384
    parser.add_argument(
        "--max_len",
        type=int,
        default="16384",
        help="Length of new_max_tokens",
    )

    args = parser.parse_args()
    # print(args)
    main(args)
