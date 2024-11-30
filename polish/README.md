<!--
 * @Author: panchang 20212121026@m.scnu.edu.cn
 * @Date: 2024-11-29 10:07:22
 * @LastEditors: panchang 20212121026@m.scnu.edu.cn
 * @LastEditTime: 2024-11-29 11:58:08
 * @FilePath: \music_generation\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# 🌟 Polish Module

> **The polish module of ToneCraft Framework**




## 📝 Module Overview

This module is designed to *polish* generated melodies within an acceptable range, making them better suited to the lyrics. It focuses on improving the *harmony* between the melody and the lyrics through a series of strategies and techniques.

The module provides the following key components:

1. **Training Data for Instruction Tuning**: 
   - The training data is used to fine-tune the model for the *masking* task of pitch sequence.
   - It utilizes the same **instruction tuning** framework as the CL2M module, specifically the **LLaMA-Factory** framework. [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)


2. **Inference and Testing Code for Different Polish Strategies**: 
   - This part of the module includes code for inferring and testing various *polish* strategies, allowing for a flexible evaluation of different approaches to melody refinement.

3. **Case Studies Analysis**:
   - The module includes several case studies to demonstrate its effectiveness:
     - **Case 1**: Derived from a real song.
     - **Case 2-4**: by the open-source music model available on [Hugging Face](https://huggingface.co/sander-wood/text-to-music). The model generates songs in **ABC notation**, which are then converted into pitch sequences for further processing and analysis.

This module is intended for improving melody-lyric alignment and testing different polish strategies to achieve more harmonious song generation.




## 📂 File Structure


```plaintext
polish/
│
├── case_study/            # Containing some cases
│   ├── case_study.ipynb   # Case analysis
│   ├── case1_sleep.json   # pitch sequeces of case1
│   ├── case2_jiangnan.json
│   ├── case3_jazz.json
│   ├── case4_italy.json
│   └── tone.json          # Dictionary for tone
│
├── data/                 # Directory for storing datasets
│   ├── alpaca_mask_short_data.json  # Training data
│   ├── mask_data.json         # Last 500 records for testing
│   └── short_pitch_data.json  # pitch sequence data 
│
├── test/                # Test code directory
│   ├── __init__.py     
│   ├── data_utils.py    # Utility functions for handling pitch data
│   ├── metrics.py       # Metrics calculation for MD
│   └── test.py          # Test for polish strategy   
│
└── README.md            # Module documentation
```



## 📊 Data Source

The training data for this project is derived from the **LMD-matched** subset of the [Lakh MIDI Dataset (LMD)](https://colinraffel.com/projects/lmd/), a collection of high-quality MIDI files that have been matched to their corresponding entries in the Million Song Dataset (MSD). This subset provides a robust alignment of MIDI and metadata, making it ideal for tasks requiring both audio and symbolic music information.

- **Original Data Source**: [LMD-matched](https://colinraffel.com/projects/lmd/)
- **Processing Steps**: The following preprocessing steps were applied to adapt the dataset for our task:
  1. **Feature Extraction**: Used the `pretty_midi` library to extract **pitch sequences**, **note durations**, and **rest durations** from MIDI files. The detailed extraction method is implemented in the [`songcomposer`](https://github.com/pjlab-songcomposer/songcomposer/blob/main/finetune/mid_to_tuple.py) project.
  2. **Abstraction**: Transformed the extracted triplets into pure pitch sequences for further processing.
  3. **Data Cleaning**: 
     - Removed sequences that were excessively long.
     - Simplified repetitive pitch patterns by limiting consecutive identical pitches to a maximum of three repetitions

The resulting custom dataset has been specifically processed to suit the requirements of this project while maintaining the integrity and alignment provided by the LMD-matched subset.  

For more information about the LMD-matched dataset, please visit the [Lakh MIDI Dataset official website](https://colinraffel.com/projects/lmd/).



### ⚠️ Data Usage Disclaimer
---
The LMD-matched subset is used in accordance with the terms and conditions outlined by the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/). The processed dataset created for this project is intended for **non-commercial research purposes only**. For more details, refer to the original dataset's terms of use.




