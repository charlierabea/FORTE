# RADICAL

> this repository is modified from https://github.com/Luodian/Otter

## Training
bash /Otter_train/0924_train_baseline.sh(or any other variation)

1. The external checkpoint folder arg should be checked
2. —run_name is the checkpoint name, need to be modified(also change the project name):
Eg:
--wandb_entity=big_data_center \
--run_name=OTTER-LLaMA7B-MED_CLIP \
--wandb_project=OTTER-LLaMA7B-MED_CLIP \

## Evaluating(1)— generate caption
bash /Otter_eval/eval-otter_baseline.sh(or any other variation)
Check the 
(1) eval.py file: change the excel path and prompt
(2) checkpoint(must be hf folders)
(3) mimicit_path: the corresponding prompt variation

If your checkpoint's not a hf folder, use this converter:
/Otter_eval/otter/converting_otter_pt_to_hf.py

python3 converting_otter_pt_to_hf.py --old_ckpt_path=/xx/Otter_checkpoints/0925_OTTER_CLIP_ABC/final_weights.pt --new_hf_path=/xx/Otter_checkpoints/0925_OTTER_CLIP_ABC_hf/ --pretrained_model_path=/xx/Otter_checkpoints/OTTER-MPT7B-Init/


## Evaluating(2)— generate BLEU/CIDEr scores
/code/0901_eval_bleu_cider.py
