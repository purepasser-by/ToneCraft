
import torch 
from modelscope import AutoModelForCausalLM,AutoTokenizer 
import os
from data_utils import convert_pitches_to_numbers,div_zone,pitch2zone,cal_zone_range
from metrics import cal_md
import json
from data_utils import msk_pitch_str
import random
import numpy as np  
import re
import csv
import itertools




# 自定义采样策略，采样三个候选项并记录每步的概率和累积概率
def custom_sampling(logits, num_samples=3, temperature=0.8, top_p=0.9):
    # 应用 temperature 和 top-p 策略
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0  # 保留至少一个 token

    # 抑制被过滤的 token
    filtered_logits = logits.clone()
    filtered_logits[:,sorted_indices[sorted_indices_to_remove]] = -torch.inf

    # 在过滤后的分布上采样三个 token
    probabilities = torch.softmax(filtered_logits, dim=-1)
    sampled_tokens = torch.multinomial(probabilities, num_samples=num_samples, replacement=False)
    
    # 获取对应 token 的概率
    sampled_probs = probabilities.gather(1, sampled_tokens)
    
    return sampled_tokens, sampled_probs



# 自定义生成函数
def custom_generate(input_ids,true_zones,zone_bound,lmt, max_new_tokens=256, eos_token_id=None):
    generated_ids = input_ids
    total_probabilities = []  # 记录每步的候选项及其概率信息

    cnt_yinfu=0
    for step in range(max_new_tokens):
        # 获取当前步的 logits
        outputs = model(generated_ids)
        logits = outputs.logits[:, -1, :]

        # 使用自定义采样策略获取3个候选项及其概率
        next_tokens, token_probs = custom_sampling(logits, num_samples=3)
        
        # 记录当前步候选项及其概率
        step_probabilities = []
        for i in range(3):
            token_id = next_tokens[0, i].item()
            prob = token_probs[0, i].item()
            step_probabilities.append((token_id, prob))
        
        total_probabilities.append(step_probabilities)

        # 在这里加限制，选择一个符合或者比较符合音区的，继续生成序列
        has_yinfu=0
        for i in range(3):
            temp_chosen = next_tokens[0, i].unsqueeze(0) #确保能选到是音符
            if eos_token_id is not None and temp_chosen == eos_token_id: #终止符号优先级最高  第一个字符都是空格
                chosen_token = temp_chosen
                break
            '''
            if step==0:
                chosen_token=torch.tensor([220]).cuda()#空格
                break
            '''
            ls=convert_pitches_to_numbers(tokenizer.decode(temp_chosen))
            if(len(ls)==0):#不是音符
                if i==2 and has_yinfu==0:
                    chosen_token=torch.tensor([220]).cuda()#空格
                    break
                continue #直接跳过  3个可选，后面肯定会遇到有音符
            has_yinfu=1
            chosen_token=temp_chosen
            chosen_pitch=ls[0] #选择的音高
            try:
                zone_id=true_zones[cnt_yinfu]#应该填这个音区
            except IndexError:
                print("IndexError，ignore this data")
            pitch_range=cal_zone_range(zone_bound,zone_id)
            if(pitch_range[0]-lmt<= chosen_pitch <=pitch_range[1]+lmt):
                #print("第{}次接受".format(i))
                cnt_yinfu+=1
                break #选到了，可以提前结束
            

        generated_ids = torch.cat([generated_ids, chosen_token.unsqueeze(0)], dim=1)

        # 如果生成了 eos_token 则终止
        if eos_token_id is not None and chosen_token == eos_token_id:
            break

    '''
    # 输出每步的候选项及其概率
    for step, candidates in enumerate(total_probabilities):
        print(f"Step {step + 1}:")
        for token_id, prob in candidates:
            token_str = tokenizer.decode([token_id])
            print(f"  Token: {token_str}, Probability: {prob:.4f}")
        
        # 计算并输出总的累积概率（仅选择第一个候选项的路径）
        if step == 0:
            cumulative_prob = total_probabilities[0][0][1]
        else:
            cumulative_prob *= total_probabilities[step][0][1]
        print(f"  Cumulative Probability up to this step: {cumulative_prob:.4f}\n")
    '''

    return generated_ids


def inference(msg,true_zones,zone_bound,lmt):
    # 将消息模板转换为模型输入格式
    prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    # 定义终止条件
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # 对 prompt 进行编码
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")


    # 使用自定义生成函数
    outputs = custom_generate(
        input_ids=input_ids,
        true_zones=true_zones,
        zone_bound=zone_bound,
        lmt=lmt,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 解码并打印生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print("\nGenerated Text:\n", generated_text)

    return generated_text




def generate_mask_msg(record):
  
    pitch = record["pitch"]

    pitch_length = len(pitch)
    
    # 根据 pitch 长度确定掩码数量
    if 6 <= pitch_length <= 8:
        num_masks = 1
    elif 9 <= pitch_length <= 11:
        num_masks = random.randint(1, 2)
    elif 12 <= pitch_length <= 14:
        num_masks = random.randint(1, 3)
    else:
        # 如果长度不在 6 到 14 之间，跳过当前记录
        #continue
        num_masks = random.randint(1, 3)

    # 生成 mask 的起始位置
    mask_index = []
    
    
    # 确保掩码起始位置不超出下标范围
    start_index = random.randint(0, pitch_length - num_masks)
    mask_index.extend(range(start_index, start_index + num_masks))

    input_str=msk_pitch_str(pitch,mask_index)

    # 定义指令和用户输入
    msg_tpl = [
        {"role": "system", "content": "你是一个专业的作曲家"},
        {"role": "instruction",
         "content": "请你根据给定的旋律，填写旋律中[M]的内容使前后连贯，注意输出的output的长度要和[M]的个数严格匹配，下面有{}个[M]".format(len(mask_index))}
    ]

    # 用户音高序列输入
    user_input = input_str
    msg = msg_tpl + [{"role": "user", "content": user_input}]

    return msg,mask_index


def generate_mask_data(record):  
    mask_record={}
    msg,mask_index=generate_mask_msg(record)
    mask_record['mask_idx']=mask_index

    #计算音区的分区  需要填写的音区
    zone_bound=div_zone(record['pitch'])
    mask_true_ls=[record['pitch'][i] for i in mask_index]
    ori_zones=[pitch2zone(p,zone_bound) for p in mask_true_ls]#旋律原本的音区
    #随机给音区  但是不能是原本的音区  （修改音区）
    #true_zones=[random.randint(1,4) for i in range(len(mask_index))]
    true_zones=[]
    for ori_z in ori_zones:
        true_z=random.randint(1,4)
        while true_z==ori_z:
            true_z=random.randint(1,4)#直到选到不一样的音区
        true_zones.append(true_z)

    mask_record['require_zones']=true_zones
    mask_record['msg']=msg

    return mask_record


# 返回和谐度   前两个参数只传修改的音区  len是整个序列
def harmony(ori_zone,require_zone,all_len):
    dif_len = len(ori_zone)
    
    score = 0.
    
    for i in range(dif_len):
        t = ori_zone[i]
        try: 
            w = require_zone[i]
        except IndexError:
            print("IndexError，ignore this data")
        

        d = abs(t-w)
        score += np.exp(-d)
    for i in range(all_len-dif_len):
        score += np.exp(0)

    score /= all_len + 1e-9

    return score


def cal_hmn_increase(data):

    # 计算 hmy 相较于 old_hmy 提高的平均百分比
    total_percentage_increase = 0
    total_increase=0
    count = 0
    old_hmy_avg=0
    new_hmy_avg=0
    md_avg=0

    for item in data:#accepted_items:
        try_times=len(item)-4
        if item['accept']==True:
            hmy = item[try_times]['hmn']
            md=item[try_times]['md']
        else:
            hmy = item['old_hmn']
            md=0
        old_hmy = item['old_hmn']
        
        if hmy is not None and old_hmy is not None and old_hmy != 0:
            old_hmy_avg+=old_hmy
            new_hmy_avg+=hmy
            md_avg+=md
            increase=hmy - old_hmy
            percentage_increase = ((hmy - old_hmy) / old_hmy) * 100
            total_increase += increase
            total_percentage_increase += percentage_increase
            count += 1

    # 计算平均百分比
    average_percentage_increase = total_percentage_increase / count if count > 0 else 0
    average_increase=total_increase / count if count > 0 else 0
    old_hmy_avg = old_hmy_avg / count if count > 0 else 0
    new_hmy_avg=new_hmy_avg / count if count > 0 else 0
    md_avg=md_avg/count if count > 0 else 0
    #print("平均提的百分比：",average_percentage_increase)
    #print("平均提的点（绝对）：",average_increase)
    return average_percentage_increase,average_increase,old_hmy_avg,new_hmy_avg,md_avg


def calculate_accept_ratio(data):

    # 计算 accept 为 true 的数量
    total_items = len(data)
    accepted_items = sum(1 for item in data if item.get('accept') == True)

    # 计算 accept 为 true 的比例
    accept_ratio = accepted_items / total_items if total_items > 0 else 0

    #print("接受比例:",accept_ratio)
    return accept_ratio




def test_polish(max_md,start_lmt,max_lmt,max_cnt):
    new_data=[]

    for num,record in enumerate(test_data[:100]):
        new_record=dict()
        cnt=0 #记录生成了多少次
        #msg,mask_index=generate_mask_msg(record)
        msg=mask_data[num]['msg']
        mask_index=mask_data[num]['mask_idx']

        new_record['pitch']=record['pitch']
        new_record['mask_idx']=mask_index

        # 兼容一下0 mask的数据
        if len(mask_index)==0:
            new_record['old_hmn']=len(record['pitch']) /(len(record['pitch']) + 1e-9)
            new_record["accept"]=True
            new_record[1]=dict()
            new_record[1]['hmn']=len(record['pitch']) /(len(record['pitch']) + 1e-9)
            new_record[1]['md']=0
            new_data.append(new_record)
            continue #跳过该数据，因为无mask



        lmt=start_lmt
        #计算音区的分区  需要填写的音区
        zone_bound=div_zone(record['pitch'])
        mask_true_ls=[record['pitch'][i] for i in mask_index]
        ori_zones=[pitch2zone(p,zone_bound) for p in mask_true_ls]#旋律原本的音区
        true_zones=mask_data[num]['require_zones']
        #计算随机mask，旧的和谐度
        new_record['old_hmn']=harmony(ori_zones,true_zones,len(record['pitch']))


        #开始多次生成，直到满足条件或到达次数
        while 1:
            if cnt>=max_cnt: 
                break
            cnt+=1
            new_record[cnt]=dict()

            output=inference(msg,true_zones,zone_bound,lmt)
            output=re.search(r'(?<=assistant)(.*)', output, re.DOTALL).group(1).strip()
            mask_ls=convert_pitches_to_numbers(output)

            new_record[cnt]['output']=mask_ls
            

            #填充mask构造回原来的序列
            ori_ls=[record['pitch']]
            polish_ls=[[i for i in record['pitch']]]

            index_error=0
            for i,idx in enumerate(mask_index):
                try: #有下标越界的错误没有解决，先不管
                    polish_ls[0][idx]=mask_ls[i]
                except IndexError:
                    print("IndexError，ignore this data")
                    index_error=1
                    break
            if index_error==1:
                continue #跳过继续
                
            new_record[cnt]['polish']=polish_ls[0]

            md=cal_md(polish_ls,ori_ls)
            new_record[cnt]['md']=md

            #计算新的和谐度
            polish_zones=[pitch2zone(p,zone_bound) for p in mask_ls]
            new_record[cnt]['hmn']=harmony(polish_zones,true_zones,len(record['pitch']))

            if md<max_md:
                new_record['accept']=True
                break

            #放宽lmt
            if cnt%2==0: #2轮加一次，每次加2
                if lmt+(cnt)/2*2<=max_lmt:
                    lmt=lmt+(cnt)/2*2


        if cnt==max_cnt:
            new_record['accept']=False

        new_data.append(new_record)
        print("第{}个完成".format(num))


    output_name="polish_md_{}_startlmt_{}_lmt_{}_msk.json".format(max_md,start_lmt,max_lmt)
    # 有需要可以将polish过程和指标写文件
    '''
    with open(output_name, "w") as outfile:
        json.dump(new_data, outfile, indent=4)
    '''

    average_percentage_increase,average_increase,old_hmy_avg,new_hmy_avg,md_avg=cal_hmn_increase(new_data)
    ac_rate=calculate_accept_ratio(new_data)

    return ac_rate,average_percentage_increase,average_increase,old_hmy_avg,new_hmy_avg,md_avg



def run_test(max_md, start_lmt, max_lmt):
    ac_rate, rate_increase, absolute_increase,old_hmy_avg,new_hmy_avg,md_avg=test_polish(max_md,start_lmt,max_lmt,5)
    return ac_rate, rate_increase, absolute_increase,old_hmy_avg,new_hmy_avg,md_avg




if __name__=="__main__":

    model_path = 'your_model_path'
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map ="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path,torch_dtype=torch.float16,device_map ="cuda") 

    
    with open("../data/short_pitch_data.json", 'r') as infile:
        data = json.load(infile)

    
    test_data=data[-500:]# 测试后500条数据，要不和训练集交叉


    # 如果没有数据，运行下面的代码先生成测试数据
    '''
    mask_data=[]
    
    for record in test_data:
        mask_record=generate_mask_data(record)
        mask_data.append(mask_record)
    

    
    with open("mask_data.json", "w") as outfile:#这个原来文件存的是从-500开始的
        json.dump(mask_data, outfile, indent=4)
    '''
    

    with open("mask_data.json", 'r') as infile:# 这个数据是从-500开始的
        mask_data = json.load(infile)
    
    
    
    # 只测试单个策略
    # max_md=1
    # max_lmt=0
    # start_lmt=0
    # max_cnt=5


    # 搜索：测试多种策略
    # 参数列表
    max_md_list =[0.8, 1.0, 1.2]
    start_lmt_list = [0, 2, 6, 10, 20]
    max_lmt_list = [0, 2, 6, 10, 20]

    # 输出CSV文件名
    output_filename = "batch_test_results.csv"

    # 创建CSV文件并写入结果
    with open(output_filename, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # 写入表头
        header = ["测试情况", "max_md", "start_lmt", "max_lmt", "ac rate", "hmn improve rate", "hmn improve value","old_hmy_avg","new_hmy_avg","md_avg"]
        csv_writer.writerow(header)

        # 生成所有参数组合并测试
        for max_md, start_lmt, max_lmt in itertools.product(max_md_list, start_lmt_list, max_lmt_list):
            if max_lmt< start_lmt:
                continue
            if start_lmt<10 and max_lmt>=10:
                continue
            
            # 定义测试情况名称
            test_case_name = f"md{max_md}_start{start_lmt}_max{max_lmt}"
            
            # 运行测试
            ac_rate, rate, absolute_increase,old_hmy_avg,new_hmy_avg,md_avg = run_test(max_md, start_lmt, max_lmt)
            print(test_case_name,"运行完毕，结果",ac_rate,"  ", rate,"  ", absolute_increase," ",old_hmy_avg," ",new_hmy_avg," ",md_avg)
            
            # 写入CSV行数据
            row = [test_case_name, max_md, start_lmt, max_lmt, ac_rate, rate, absolute_increase,old_hmy_avg,new_hmy_avg,md_avg]
            csv_writer.writerow(row)

    print(f"测试结果已写入文件: {output_filename}")