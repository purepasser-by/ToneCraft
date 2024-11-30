from modelscope import AutoTokenizer,AutoModelForCausalLM
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from pkgs.canto.utils import show_lyrics,filter_tokens,tokens2digits,kr_tokens2digits,usrtext2tokens,kr_usrtext2tokens
import os
import random

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name','-n',type=str,default='qwen2_tc')
    parser.add_argument('--res_dir','-r',type=str,default='./results')
    parser.add_argument('--data_num','-d',type=int,default=10) # 随机抽选多少数据
    parser.add_argument('--gen_num','-g',type=int,default=10) # 每条数据生成几次
    return parser.parse_args()


class Pipeline:
    def __init__(self, model_path : str):
        self.model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map ="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,torch_dtype=torch.float16,device_map ="auto")
        self.template_instruction = "根据给定的Pitches，生成与之适配且相同长度的歌词，每个Pitch对应一个中文字符。"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.is_qwen = False
        if "qwen" in model_path:
            self.is_qwen = True

    def prepare_msg(
            self,
            relative_pitches : str, 
            previous_lyrics : list[str] = None, 
            composition_requirements : str = None
        ) -> list[dict]:

        instruction = self.template_instruction
        if previous_lyrics:
            rhyme = previous_lyrics[0][-1]
            instruction += "此外，生成的歌词要与给定的Previous lyrics连贯。" + f"Previous lyrics : {str(previous_lyrics)}\n"
            instruction += "同时，生成的最后一个中文字符需要和给定的Rhyme押韵。" + f"Rhyme: {rhyme}\n"
        if composition_requirements:
            instruction += composition_requirements + "\n"

        melody = usrtext2tokens(relative_pitches) if not self.is_qwen else kr_usrtext2tokens(relative_pitches)
        msg = [
            {"role": "instruction","content":instruction + f"[Character Nums: {len(relative_pitches)}]"},
            {"role":"input","content": "Pitches: " + melody}
        ]
        return msg

    def generate(
            self,
            relative_pitches_segs : str, 
            previous_lyrics : str = None, 
            composition_requirements : str = None,
            streamer = None
        ) -> str :
        relative_pitches_segs = relative_pitches_segs.replace(" ","").split("|")
        if previous_lyrics is not None:
            previous_lyrics = previous_lyrics.replace(" ","").split("|")
        res = ""
        for relative_pitches in relative_pitches_segs:
            prompt = self.tokenizer.apply_chat_template(
                self.prepare_msg(relative_pitches,previous_lyrics,composition_requirements),
                tokenize=False,
                add_generation_prompt=True
            )
            _res = self._generate(prompt,len(relative_pitches),streamer)
            if previous_lyrics is None:
                previous_lyrics = []
            previous_lyrics.append(_res)
            if len(previous_lyrics) > 3:
                previous_lyrics = previous_lyrics[-3:]
            res += _res + "\n"

        return res
    
    def _generate(self, prompt : str, max_new_tokens : int = None, streamer = None) -> str:

        inputs =self.tokenizer([prompt], return_tensors="pt").to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        input_ids =inputs["input_ids"].to(self.device)
        
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            max_new_tokens= max_new_tokens if max_new_tokens else 256,
            temperature=0.9,
            do_sample=True,
            top_p= 0.6,
            repetition_penalty= 1.1,
            top_k = 0,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer = streamer
        )

        res = outputs[0][len(inputs[0]):].view(outputs.size(0),-1).contiguous()
        res = self.tokenizer.batch_decode(res,skip_special_tokens=True)[0]
        return res


# Generate Results

if __name__ == '__main__':
    args = parse_config()
    assert os.path.exists(args.res_dir) 

    # 加载测试数据
    if "qwen" in args.model_name:
        tokens2digits = kr_tokens2digits
        usrtext2tokens = kr_usrtext2tokens
        test_data_path = "./data/canto_hmn_gen_kr.json"
    else:
        test_data_path = "./data/canto_hmn_gen.json"
    with open(test_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_data = random.sample(
        [(entry["instruction"], entry["input"], entry["output"]) for entry in data],
        args.data_num
    )
    test_data = random.sample(
        [(entry["instruction"], entry["input"], entry["output"]) for entry in data if len(entry["input"]) > 15],
        args.data_num
    )

    # 加载 pipeline
    model_path = './models/' + args.model_name
    pipeline = Pipeline(model_path)


    # 对 args.data_num 条数据， 每条生成 args.gen_num 次
    results_output = []
    for (instruction, melody, labels) in tqdm(test_data, desc="处理测试集的所有旋律"):
        melody_ = melody
        melody = tokens2digits(melody)
        new_entry = {
            "melody": melody,
            "inputs": melody_,
            "labels": labels,
            "lyrics": [],
            "avg_cnt": 0,
        }
        avg_cnt = 0
        for i in tqdm(range(args.gen_num), desc="为每段旋律生成歌词", leave=False):
            cnt = 0
            while True:
                res = pipeline.generate(melody).strip()
                cnt += 1
                if cnt > 6:
                    avg_cnt += 6
                    break
                elif len(melody) != len(res):
                    tqdm.write(f"Cnt: {cnt}, {melody}, {res}")
                else :
                    new_entry["lyrics"].append(res)
                    avg_cnt += cnt
                    break

        avg_cnt /= args.gen_num
        new_entry["avg_cnt"] = avg_cnt

        results_output.append(new_entry)

    output_path = args.res_dir + "/" + args.model_name 
    with open(output_path + ".json", 'w', encoding='utf-8') as f:
        json.dump(results_output, f, ensure_ascii=False, indent=2)

