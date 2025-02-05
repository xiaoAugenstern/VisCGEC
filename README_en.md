# VisCGEC: Benchmarking the Visual Chinese Grammatical Error Correction

English | [简体中文](./README.md)


VisCGEC is a novel benchmark dataset of visual Chinese grammatical error correction consisting of 2, 451 images collected from real-world CFLs’ handwritten texts. 


![](./UnifiedGEC.jpg)


## Dataset Overview
VisCGEC is the first dataset to focus on visual Chinese grammatical error correction, combining both textual and image modalities. The dataset is designed to reflect real-world scenarios where CFL learners often make phonological and visual errors when writing Chinese characters. 

Each image in the dataset is annotated with:
- Recognized texts (with errors). 
- Corrected texts (grammatically correct).

### Key Features of VisCGEC:
- Diverse Writing Styles: Handwritten texts from 303 CFL learners with varying proficiency levels (HSK4 to HSK6).
- Broad Error Types: Includes grammatical errors, misspelled characters, and faked characters (non-existent characters created due to writing errors).
- Multimodal Data: Combines visual (handwritten images) and textual (recognized and corrected texts) modalities.


## Dataset Statistics

| Property            | Train | Dev  | Test |
|---------------------|-------|------|------|
| # Images           | 1960  | 245  | 246  |
| Avg. Source Length | 24.27 | 25.00 | 23.18 |
| Avg. Target Length | 24.57 | 25.52 | 23.67 |
| # Edits per Sentence | 1.78  | 1.78 | 1.84 |

## Baseline Approaches
We propose two baseline frameworks for the VisCGEC task:
1. Two-Stage Pipeline:
    - Stage 1: Character Recognition: Uses OCR or CLIP-based methods to recognize characters from handwritten images. 
    - Stage 2: Error Correction: Applies Seq2Seq, Seq2Edit, or fine-tuned LLMs (e.g., Qwen) to correct grammatical errors.

2. End-to-End System:
    - Utilizes Multimodal Large Language Models (MLLMs) like GPT-4 to directly correct errors in handwritten images.


## Performance of different baseline approaches on the VisCGEC




