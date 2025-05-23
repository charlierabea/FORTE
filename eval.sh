export PYHTONPATH="./evaluation/" # Change this path to your local path of FORTE/evaluation/ directory, where python will look for additional module. Necessary for correctly importing modules from evaluation/ 

#This code capture the report output during the optimization process and record the report to an excel file (which should be specified in /pipeline/train/eval.py
accelerate launch --config_file=./evaluation/pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
./evaluation/pipeline/train/eval.py \
--pretrained_model_name_or_path="./checkpoints/OTTER_CLIP_BRAINGPT_hf/" \
--mimicit_path="./data/CQ500p_instruction.json" \
--images_path="./data/CQ500p.json" \
--batch_size=1 \
--warmup_steps_ratio=0.01 \
--workers=1 \
