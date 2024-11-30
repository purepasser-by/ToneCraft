import pretty_midi
import re
import numpy as np


def ls_2_region(ls):
    str="<bor> "
    mp={1:'<p1>',2:'<p2>',3:'<p3>',4:'<p4>'}
    for i in ls:
        str=str+mp[i]+' '
    str=str+'<eor>'
    return str


def msk_pitch_str(pitch,mask_index):
    user_input = ""
    for idx, item in enumerate(pitch):
        if idx in mask_index:
            user_input = user_input + ' [M]'
        else:
            user_input = user_input + ' <' + pretty_midi.note_number_to_name(item) + '>'
    return user_input



def convert_pitch_to_number(pitch_str):
    pitch_str = pitch_str.strip("<>").upper()
    try:
        # 转换音符字符串为 MIDI 音高
        midi_note = pretty_midi.note_name_to_number(pitch_str)
        return midi_note
    except ValueError:
        # 如果不是有效音符，返回 None 或抛出自定义异常
        print(f"Warning: '{pitch_str}' is not a valid note.")
        return None  # 或者返回一个默认值，例如 -1


def convert_pitches_to_numbers(pitch_string):
    # 使用正则表达式从字符串中提取所有的音符（<音符>）
    pitches = re.findall(r'<([^>]+)>', pitch_string)

    # 将每个音符转换为数字音高
    pitch_numbers = [convert_pitch_to_number(pitch) for pitch in pitches]

    # 过滤掉转换失败的音符（返回 None 的情况）
    return [num for num in pitch_numbers if num is not None]



def div_zone(pitches):
    pp = [0,1,2,3,4]
    inters = list(np.linspace(min(pitches), max(pitches), 5))
    for i in range(len(inters)):
        pp[i] = inters[i]
    return pp

def cal_zone_range(pp,id):
    if id==1:
        return [-100,pp[1]]
    elif id==2:
        return [pp[1],pp[2]]
    elif id==3:
        return [pp[2],pp[3]]
    else:
        return [pp[3],999]

def pitch2zone(pitch,pp):
    
    x=pitch
    if x>=pp[3]:
        return 4
    elif x>=pp[2]:
        return 3
    elif x>=pp[1]:
        return 2
    else:
        return 1



