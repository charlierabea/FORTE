export PYTHONPATH="/home/chengyili/project/CT-CLIP/Otter_original:$PYTHONPATH"

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/0830_eval.py \
# --pretrained_model_name_or_path="/home/chengyili/project/CT-CLIP/Otter/checkpoints/OTTER-LLaMA7B_0828_CLIP_hf/" \
# --eval_path="/local2/chengyili/data/output/eval.json" \
# --eval_instruction_path="/local2/chengyili/data/output/eval_instruction.json" \
# --batch_size=1 \

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/0830_eval.py \
--pretrained_model_name_or_path="checkpoints/OTTER-LLaMA7B-MED_CLIP_hf/" \
--mimicit_path="/local2/chengyili/data/output/MED_instruction.json" \
--images_path="/local2/chengyili/data/output/MED.json" \
--batch_size=1 \
--warmup_steps_ratio=0.01 \
# --report_to_wandb \
# --wandb_entity=big_data_center \
# --run_name=OTTER-0828_eval \
# --wandb_project=OTTER-LLaMA7B-0831_eval \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
