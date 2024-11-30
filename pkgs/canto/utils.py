import numpy as np
import json
import zhconv
import os
import re
from scipy.stats import spearmanr
import warnings


def sayhi():
    print("hi")

# 返回某个字对应的声调
def tone(x : str) -> int:
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    fp = utils_dir + "/tone.json"
    with open(fp,'r',encoding='utf-8') as jf:
        tone_js = json.load(jf)
    if x == '.' or x == ',' or x == '。' or x == '，': 
        return 0
    value = tone_js.get(x)
    if value :
        return int(value)
    else:
        x = zhconv.convert(x,"zh-hant")
        value = tone_js.get(x)
        if value :
            return int(value)
        else:
            return 0
        


# 返回某个声调对应的音高
def tone2pitch(x : int) -> int:
    tone_md = {0: 0, 1: 4, 2:4, 3: 3, 4: 1, 5: 3, 6: 2,-1: 0}
    return tone_md[x]


# 去掉生成的特殊token
def filter_tokens(s : str) -> str:
    s = s.replace("<|bar|>","").replace("<|eot_id|>","").replace("<|cma|>","").replace(" ","").replace("\n","")
    s = s.replace("|","").replace("<im_end>","").replace("<|im_end|>","").replace("，","").replace("。","").replace(",","").replace(".","")
    return s


# 展示歌词结果
def show_lyrics(s : str) -> str:
    s = s.replace(" ","").replace("<|cma|>",",").replace("<|bar|>",".").replace("<|eot_id|>","")
    return s


# 返回和谐度
def harmony(real_pitches : str, words : str) -> float:
    real_pitches = real_pitches.replace(" ","").replace("|","").strip()
    words = filter_tokens(words).replace(",","").replace(".","").strip()

    assert len(real_pitches) == len(words), f"Length mismatch: len(tones) = {len(real_pitches)}, len(words) = {len(words)}"
    length = len(real_pitches)
    
    score = 0.
    for i in range(length):
        t = int(real_pitches[i])
        w = words[i]

        d = abs(t-tone2pitch(tone(w)))
        score += np.exp(-d)
    score /= length + 1e-9

    return score



# 返回趋势一致性
def consistency(real_pitches : str, words : str) -> float:
    real_pitches = real_pitches.replace(" ","").replace("|","").strip()
    words = filter_tokens(words).replace(",","").replace(".","").strip()
    assert len(real_pitches) == len(words), f"Length mismatch: len(real_pitches) = {len(real_pitches)}, len(words) = {len(words)}"

    real_pitches = [int(p) for p in real_pitches]
    words = [tone2pitch(tone(w)) for w in words]
    corr , _ = spearmanr(real_pitches,words)
    if np.isnan(corr):
        warnings.warn(f"Spearman correlation resulted in NaN for real_pitches = {real_pitches} and words = {words}")
        return None
    else:
        return corr
    


# 判断给定的字符串是否包含两个字及以上的中文字符
def has_two_chinese_chars(s: str) -> bool:
    chinese_characters = re.findall(r'[\u4e00-\u9fa5]', s)
    return len(chinese_characters) >= 2


'''
    将用户的音区输入转换为token
    " 233 234 | 3342 | "   -->  "<p2><p3><p3><p2><p3><p4><|bar|><p3><p3><p4><p2><|bar|>"
'''
def usrtext2tokens(text : str) -> str:
    return text.strip().replace("4","<p4>").replace("3","<p3>").replace("2","<p2>") \
               .replace("1","<p1>").replace("0","<p0>").replace("|","<|bar|>").replace(" ","")

def kr_usrtext2tokens(text : str) -> str:
    return text.strip().replace("4","사").replace("3","삼").replace("2","이") \
               .replace("1","일").replace("0","영").replace("|","<|bar|>").replace(" ","")


'''
    将token转换为纯数字序列
    "<p2><p3><p3><p2><p3><p4><|bar|><p3><p3><p4><p2><|bar|>"   -->  "2332343342"
'''
def tokens2digits(text : str) -> str:
    return text.strip().replace("<|bar|>","").replace("<p4>","4").replace("<p3>","3").replace("<p2>","2") \
               .replace("<p1>","1").replace("<p0>","0").replace("<|endoftext|>","").replace("Pitches: ","")

def kr_tokens2digits(text : str) -> str:
    return text.strip().replace("<|bar|>","").replace("사","4").replace("삼","3").replace("이","2") \
               .replace("일","1").replace("영","0").replace("<|endoftext|>","").replace("Pitches: ","")
