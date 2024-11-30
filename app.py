import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread,Event
import io
from PIL import Image
from pkgs.canto.utils import usrtext2tokens,show_lyrics,filter_tokens,harmony,tone,tone2pitch,kr_tokens2digits,kr_usrtext2tokens
import matplotlib.pyplot as plt
import time
from inference import Pipeline
import argparse

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name','-n',type=str,default='qwen2_tc')
    return parser.parse_args()


stop_event = Event()

def stop_generation():
    global stop_event
    stop_event.set() 


def matching_curve(y_true : str, words : str):
    y_true = [int(p) for p in y_true]
    words_pitches = [tone2pitch(tone(w)) for w in words]
    x = list(range(len(y_true)))

    plt.figure(figsize=(8,4))
    plt.plot(x, y_true, marker='o', label='Melody Pitch', color='blue')
    plt.plot(x, words_pitches, marker='s', label='Lyric Pitch', color='orange')

    plt.title(f'Lyrics-Melody Matching Curve of Relative Pitch')
    plt.ylabel('Relative Pitch')
    plt.legend()
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()  
    buf.seek(0)


    image = Image.open(buf)
    return image  

def generate(
        relative_pitches_segs : str, 
        previous_lyrics : str = None, 
        composition_requirements : str = None
    ):
    
    if previous_lyrics.strip() == "":
        previous_lyrics = None
    if composition_requirements.strip() == "":
        composition_requirements = None

    pitches = filter_tokens(relative_pitches_segs) 

    streamer = TextIteratorStreamer(pipeline.tokenizer, skip_prompt=True)
    
    
    generated_text = ''
    cnt = 0
    while True:
        relative_pitches_segs = relative_pitches_segs.replace(" ","").split("|")

        if previous_lyrics is not None:
            previous_lyrics = previous_lyrics.replace(" ","").split("|")
        for relative_pitches in relative_pitches_segs:
            _res = ""

            prompt = pipeline.tokenizer.apply_chat_template(
                pipeline.prepare_msg(relative_pitches,previous_lyrics,composition_requirements),
                tokenize=False,
                add_generation_prompt=True
            )
            
            generation_kwargs = dict(
                prompt = prompt, 
                max_new_tokens = len(relative_pitches), 
                streamer=streamer, 
            )
            thread = Thread(
                target = pipeline._generate,
                kwargs=  generation_kwargs
            )
            thread.start()

            for new_text in streamer:
                generated_text += new_text
                _res += new_text
                yield generated_text, None, None
            
            generated_text += " | "
            yield generated_text,None,None,None

            if previous_lyrics is None:
                previous_lyrics = []
            previous_lyrics.append(_res)

        
        generated_text = generated_text.replace("instruction", "").rstrip("| ")

        
        words = filter_tokens(generated_text)
        print(len(pitches),pitches)
        print(len(words),words)
    
        if len(pitches) == len(words):
            lyrics = show_lyrics(generated_text)
            yield gr.update(value=lyrics), \
            gr.update(value = round(harmony(pitches,words),4)), \
            matching_curve(pitches,words)
            break

        elif cnt > 5:
            break
        cnt += 1
        time.sleep(0.67)

if __name__ == "__main__":
    args = parse_config()
    model_path = "./models/" + args.model_name
    if "qwen" in args.model_name:
        tokens2digits = kr_tokens2digits
        usrtext2tokens = kr_usrtext2tokens

    pipeline = Pipeline(model_path)

    with gr.Blocks() as demo:
        gr.Markdown("# Cantonese Melody to Lyrics Generation(CM2L) with ToneCraft") 
        gr.Markdown("<div style='text-align:center; color:white;'>\"宏愿纵未了 奋斗总不太晚\"</div>", elem_id="footer")

        # 输入框
        with gr.Row():
            # e.g. 33233 234 43213 | 332223 3342
            relative_pitches_segs = gr.Textbox(label="Relative Pitches", placeholder="like: 123 | 3321 ")


            # e.g. 宁愿滞留在此处 | 宁愿叫时间终止  
            previous_lyrics = gr.Textbox(label="Previous Lyrics", placeholder="(Optional)")

            # e.g. 生成让人感到快乐的歌词
            composition_requirements = gr.Textbox(label="Composition Requirements", placeholder="(Optional)")

        # 输出框
        with gr.Row():
            lyrics_output = gr.Textbox(label="Generated Lyrics")
            harmony_output = gr.Textbox(label="Harmony")

        with gr.Row():

            image_output = gr.Image(type="pil", label="Matching Curve") 


        submit_button = gr.Button("Submit")

        submit_button.click(lambda: stop_event.clear(), None, None) 

        submit_button.click(generate, inputs=[relative_pitches_segs,previous_lyrics,composition_requirements], outputs=[lyrics_output, harmony_output, image_output])


        stop_button = gr.Button("Stop")

        stop_button.click(stop_generation, None, None)  


    # 启动接口
    demo.launch(server_name="127.0.0.1", server_port=7870)
