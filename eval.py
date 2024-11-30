'''
(330243 | 33224332 |) (Canto)
{441234 | 44223442 |} (Pitch)
故事的小黄花 从出生那年就飘着
'''

import os
from modelscope import AutoTokenizer,AutoModelForCausalLM
from pkgs.canto.utils import harmony,tone,tone2pitch,filter_tokens,consistency
import json
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm



def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name','-n',type=str,default='qwen2_tc')
    parser.add_argument('--res_dir','-r',type=str,default='./results')
    parser.add_argument('-hmn', action='store_true',help='only harmony')

    return parser.parse_args()

# 和谐度
# harmony
def eval_harmony(melody : str, lyrics : list) -> float:
    _harmony = 0.
    for lyric in lyrics:
        _harmony += harmony(melody,lyric)
    return _harmony / len(lyrics)



# trend_consistency
def eval_consistency(melody : str, lyrics : list) -> float:
    _trend_consistency = 0.
    valid_num = len(lyrics)
    for lyric in lyrics:
        if consistency(melody,lyric) is None:
            valid_num -= 1
        else:
            _trend_consistency += consistency(melody,lyric)
    if valid_num == 0:
        return 0
    else:
        return _trend_consistency / valid_num



# 多样性
# variation(distinct)
def eval_variation(lyrics : list) -> float:
    embeddings = embedding_model.encode(lyrics, show_progress_bar = False) # 禁用进度条 烦死了
    similarity_matrix = cosine_similarity(embeddings)
    average_similarity = np.mean(similarity_matrix)
    min_similarity = np.min(similarity_matrix[np.nonzero(similarity_matrix)])
    return average_similarity, min_similarity


# mad1 mad2 mid1 mid2
def eval_macro_micro_dist(lyrics : list) -> float:
    d1_ = 0.
    d2_ = 0.
    ugrams_ = []
    bigrams_ = []
    for lyric in lyrics:
        ugrams = [w for w in lyric]
        bigrams = []
        for bi in range(len(ugrams) - 1):
            bigrams.append(ugrams[bi] + ugrams[bi+1])

        if(len(bigrams)==0):
            print(lyric)
        d1_ += len(set(ugrams)) / float(len(ugrams))
        d2_ += len(set(bigrams)) / float(len(bigrams))

        ugrams_ += ugrams
        bigrams_ += bigrams

    macro_dist1 = d1_ / len(lyrics)
    macro_dist2 = d2_ / len(lyrics)
    micro_dist1 = len(set(ugrams_)) / float(len(ugrams_))
    micro_dist2 = len(set(bigrams_)) / float(len(bigrams_))
    return macro_dist1,macro_dist2,micro_dist1,micro_dist2


if __name__ == '__main__':


    args = parse_config()
    assert os.path.exists(args.res_dir) 

    model_path = './models/' + args.model_name

    embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")

    eval_path = args.res_dir + "/" + args.model_name 
    with open( eval_path + '.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    metrics = {
        "harmony" : 0. ,
        "consistency" : 0.,
        "avg_sim" : 0. ,
        "min_sim" : 0. ,
        "macro_dist1": 0. ,
        "macro_dist2": 0. , 
        "micro_dist1": 0. ,
        "micro_dist2" : 0. ,
        "cnt" : 0. ,
    }
    # 仅评估 harmony
    if args.hmn:
        for entry in tqdm(data,desc="Evaling Data"):
            melody = entry['melody']

            lyrics = [filter_tokens(lyric) for lyric in entry['lyrics']]
            _harmony  = eval_harmony(melody,lyrics)

            metrics["harmony"] += _harmony

        metrics["harmony"] /= len(data)

        print(f'harmony: {metrics["harmony"]:.4f} ')

    else:
        for entry in tqdm(data,desc="Evaling Data"):
            melody = entry['melody']
            labels = entry['labels']
            lyrics = [filter_tokens(lyric) for lyric in entry['lyrics']]
            _harmony = 0.
            _avg_sim = 0.
            _min_sim = 0.
            _cnt = entry['avg_cnt']
            _harmony  = eval_harmony(melody,lyrics)
            _consistency = eval_consistency(melody,lyrics)
            
            _avg_sim,_min_sim = eval_variation(lyrics)
            _macro_dist1, _macro_dist2, _micro_dist1, _micro_dist2 = eval_macro_micro_dist(lyrics)
            metrics["harmony"] += _harmony
            metrics["consistency"] += _consistency
            metrics["avg_sim"] += _avg_sim
            metrics["min_sim"] += _min_sim
            metrics["cnt"] += _cnt
            metrics["macro_dist1"] +=_macro_dist1
            metrics["macro_dist2"] += _macro_dist2
            metrics["micro_dist1"] +=_micro_dist1
            metrics["micro_dist2"] += _micro_dist2

        for key in metrics:
            metrics[key] /= len(data)

        for key in metrics:
            print(f"{key}: {metrics[key]:.4f}, ")

        with open(eval_path + ".txt", "w") as file:
            for key in metrics:
                file.write(f"{key}: {metrics[key]:.4f},\n")
