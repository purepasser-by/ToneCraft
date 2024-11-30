# ToneCraft

Our framework, ToneCraft, is capable of fully automating the Cantonese Melody to Lyric (CM2L) task and subsequent polishing. In this repository, **Qwen2-7B-Instruct** is used as an example. Let's get started!

## Essential Requirements

```bash
transformers == 4.45.2 
llamafactory == 0.9.1 
```

## Datasets
We have provided the training data used, excluding [BELLE](https://huggingface.co/datasets/BelleGroup/train_2M_CN/tree/main), in the **data** folder. So it is recommended to manually put *train_2M_CN.json* into the **data** folder

## Modified Files 


**modeling_assets**: Please ensure that *qwen_pitch_mapping.pt* is located in the same directory as *modeling_qwen2.py* in the transformers library.
And replace the remaining files with their corresponding counterparts in the original model(Qwen2) on a one-to-one basis.

**adapter.py**: We have made modifications to the /model/adapter.py file in the llamafactory library. Please ensure this file is also replaced.


**configuration_utils.py**: When training the model, the transformers library will by default print the model parameters, including some newly added attributes, which may take up a significant portion of the terminal output. If necessary, please replace this behavior as well.

## Training
Before training, you may need to download the model weight files for Qwen2-7B-Instruct.
```
modelscope download --model Qwen/Qwen2-7B-Instruct --local_dir /your_path/modelscope/qwen2_7b
```

Subsequently, we use the llamafactory library to perform LoRA training.
```
llamafactory-cli train train.yaml
```
After training is complete, you will need to merge the model weights.
```
llamafactory-cli export merge.yaml
```

## Running
Finally, run the web-based demo.
```
python app.py
```


## Polish
If you wish to further refine the output, please proceed to the **polish** directory for the corresponding operations.

## GPU Requirements
A single NVIDIA GeForce RTX 3090(24G).
