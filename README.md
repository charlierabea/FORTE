# Towards a Holistic Framework for Multimodal Large Language Models in Three-dimensional Brain CT Report Generation
<p align="center" width="100%">
<img src="overview.png"  width="80%" height="80%">
</p>
[Paper Link](https://arxiv.org/abs/2407.02235)

## Code
> this repository is modified from https://github.com/Luodian/Otter

## Set-up

1. Compare cuda version returned by nvidia-smi and nvcc --version. They need to match. Or at least, the version get by nvcc --version should be <= the version get by nvidia-smi.
2. Install the pytorch that matches your cuda version. (e.g. cuda 11.7 torch 2.0.0). We have successfully run this code on cuda 11.1 torch 1.10.1 and cuda 11.7 torch 2.0.0. You can refer to PyTorch's documentation, [Latest](https://pytorch.org/) or [Previous](https://pytorch.org/get-started/previous-versions/).
3. You may install via `conda env create -f environment.yml`. Especially to make sure the `transformers>=4.28.0`, `accelerate>=0.18.0`.

## Data preparation
Please refer to https://github.com/Luodian/Otter/issues/234#issuecomment-1665564520 to see the instructions from the original author of Otter
The directory for the data formation is at /MIIT/mimic-it/convert-it/, and the mode is ## Spot The Difference (Subtle Difference Version) 

<hints>: "short_name" should be specified at /MIIT/mimic-it/convert-it/datasets/change.py

## Multi-image instruction-tuning (MIIT)
The MIIT baseline checkpoint is at [luodian/OTTER-MPT7B-Init](https://huggingface.co/luodian/OTTER-MPT7B-Init)]
```
bash /xx/MIIT/train.sh
```

1. The external checkpoint folder arg should be checked
2. —run_name is the checkpoint name, need to be modified (also change the project name):
Eg:
--wandb_entity=TPEVGH_big_data_center \
--run_name=OTTER-LLaMA7B-MED_CLIP \
--wandb_project=OTTER-LLaMA7B-MED_CLIP \

## Evaluation
### 1. Generate reports
```
bash /xx/evaluation/eval.sh
```
Check the 
(1) eval.py file: change the excel path and instruction
(2) checkpoint(must be hf folders)
(3) mimicit_path: the corresponding prompt variation

If your checkpoint's not a "hf" folder, use this converter:

/xx/evaluation/otter/converting_otter_pt_to_hf.py
```
python3 converting_otter_pt_to_hf.py --old_ckpt_path=/xx/checkpoints/{checkpoint_name}/final_weights.pt --new_hf_path=/xx/checkpoints/{checkpoint_name}_hf/ --pretrained_model_path=/xx/checkpoints/OTTER-MPT7B-Init/
```

Our instruction-tuned model can be downloaded at [https://drive.google.com/drive/folders/1hBMpnCy9NPuEzjZJtzDByJCk5vLoDAyK?usp=drive_link]
The CQ500 external validation dataset can be requested at [http://headctstudy.qure.ai/#dataset]

### 2. Automatic Evaluation
```
python3 /xx/evaluation/automatic_evaluation.py
```

### 3. Sentence pairing and Aggregation
```
python3 /xx/evaluation/sentence_pairing.py
```

### 4. FORTE Evaluation
```
python3 /xx/evaluation/FORTE.py
```

### 5. Negation removal
```
python3 /xx/evaluation/Negation_removal.py
```
