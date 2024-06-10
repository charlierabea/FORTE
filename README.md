# Towards a Holistic Framework for Multimodal Large Language Models in Three-dimensional Brain CT Report Generation
<p align="center" width="100%">
<img src="overview.png"  width="80%" height="80%">
</p>
Multi-modal large language models (MLLMs) have been given free rein to explore exciting medical applications with a primary focus on radiology report generation. Nevertheless, the preliminary MLLM successful attempts in 2D medical image-text pair captioning are incompetent to reflect the real-world diagnostic challenge in the volumetric 3D anatomy. Toward deploying MLLM for more applicable diagnostic context, we noticed that the (1) scarcity of 3D image training dataset, (2) the direct use of undifferentiated foundation MLLMs, and (3) the lack of pertinent caption evaluation metrics were independent domain-specific constraints that integratively hobbles the iteration of next-generation medical MLLM research. In this regard, this study collected a 3D-BrainCT dataset (18,885 text-scan pairs) and applied clinical visual instruction tuning (CVIT) to train volumetric anatomy-sensible BrainGPT models to generate radiology-adherent 3D brain CT reports. Statistically, our BrainGPT model scored BLEU-1 = 44.35, BLEU-4 = 20.38, METEOR = 30.13, ROUGE-L = 47.6, and CIDEr-R = 211.77 during internal testing and demonstrated an accuracy of 0.91 in captioning midline shifts on the external validation CQ500 dataset. By further inspecting the captioned report, we reported that the traditional metrics appeared to measure only the surface text similarity and failed to gauge the information density of the diagnostic purpose. To close this gap, we proposed a novel Feature-Oriented Radiology Task Evaluation (FORTE) to estimate the clinical relevance (lesion feature and landmarks) of the report. Notably, the BrainGPT model scored an average FORTE 0.71 F1-score (degree=0.661; landmark=0.706; feature=0.693, and impression=0.779). To demonstrate that BrainGPT models possess objective readiness to generate human-like radiology reports, we conducted a Turing test that enrolled 11 physician evaluators, and around 74% of the BrainGPT-generated captions were indistinguishable from those written by humans. While various computational intelligence researchers have advocated the avant-garde MLLM applications, our work embodies a holistic framework that showcased the first-hand experience of curating a 3D brain CT dataset, fine-tuning anatomy-sensible language models, and proposing robust radiology evaluation metrics. We deemed that the adventure of docking MLLM for 3D brain CT report generation may unfold new MLLM applications at the forefront of human-machine collaborated modern healthcare.

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
2. —run_name is the checkpoint name, need to be modified(also change the project name):
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
(1) eval.py file: change the excel path and prompt
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
