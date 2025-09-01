# Causally Graph-Guided Counterfactual Analysis to  Biomedical Named Entity Recognition
Codes for our paper *Causally Graph-Guided Counterfactual Analysis to  Biomedical Named Entity Recognition*

<!-- *Visual instruction tuning towards large language and vision models with GPT-4 level capabilities.*-->

## Overview
<p align="center">
    <a> <img src=imgs/overview.png width="100%"> </a>
</p>
Overview of the Causal Graph-Guided Counterfactual Analysis (CGCA) Method. This figure uses examples from the bc4chemd dataset. The red arrow represents the counterfactual analysis path, and the red hollow arrows point to two counterfactual analysis strategies respectively. The sections enclosed by black dashed lines represent the training paths where counterfactual analysis was not utilized. The areas outlined with red dashed lines denote the paths that employed two types of counterfactual analysis methods. Additionally, the sections enclosed by orange dashed lines correspond to the Consistency Constraints module.

## Contents
- [Install](#install)
- [LLMs](#LLMs)
- [Dataset](#dataset)
- [Train](#train)

## Install
Mostly refer to fire-fly installation
1. Clone this repository and navigate to project folder

2. Install Package
```Shell
conda create -n CGCA python=3.10 -y
conda activate CGCA
pip install --upgrade pip  # enable PEP 660 support
pip install -r requirements.txt
```

## LLMs
In our study, we use [ Qwen2.5 (3B)](https://huggingface.co/Qwen/Qwen2.5-3B), [LLaMA2 (7B)](https://huggingface.co/meta-llama/Llama-2-7b) , and [LLaMA3 (8B)](https://huggingface.co/meta-llama/Meta-Llama-3-8B) for our experiments. These models cover the parameter ranges commonly employed in LLMs. LLaMA3 represents an improvement over LLaMA2, which is achieved through the use of expanded pre-training data and an augmented vocabulary.

## dataset
We conducted experiments on eight datasets of four tasks, all of which are public datasets. The detailed information is shown in the table:

| TASK  | Name  | Link  |
|-----------|-----------|-----------|
| ID | bc2gm | https://huggingface.co/datasets/spyysalo/bc2gm_corpus |
| ID | bc4chemd | https://huggingface.co/datasets/chintagunta85/bc4chemd |
| ID | genia | https://huggingface.co/datasets/Rosenberg/genia |
| ID | anatem  | https://huggingface.co/datasets/bigbio/anat_em |
| ID | ncbi | https://huggingface.co/datasets/ncbi/ncbi_disease |
| OOD | bc5cdr | https://huggingface.co/datasets/bigbio/bc5cdr |
| OOD | jnlpba | https://huggingface.co/datasets/jnlpba/jnlpba |
| OOD | biored | https://huggingface.co/datasets/bigbio/biored |


## Train
```Shell
CUDA_VISIBLE_DEVICES=2 python train_qlora.py 
    --output_dir=${OUTPUT_PATH},
    --model_name_or_path=${Model_NAME},
    --train_file=${TRAIN_FILE},
    --eval_file=${EVAL_FILE},
    --num_train_epochs=5,
    --per_device_train_batch_size=8,
    --per_device_eval_batch_size=32,
    --gradient_accumulation_steps=2,
    --learning_rate=2e-4,
    --max_seq_length=1024,
    --logging_steps=10,
    --evaluation_strategy=epoch,             
    --save_total_limit=10,
    --lr_scheduler_type=cosine,
    --warmup_ratio=0.04,
    --lora_rank=64,
    --lora_alpha=16,
    --lora_dropout=0.05,
    --task=${TASK_NAME},

    --gradient_checkpointing=true,
    --disable_tqdm=false,
    --optim=paged_adamw_32bit,
    --seed=42,
    --fp16=true,
    --report_to=tensorboard,
    --dataloader_num_workers=10,
    --save_strategy=epoch,
    --weight_decay=0.01,
    --max_grad_norm=0.3,
    --remove_unused_columns=false
```
