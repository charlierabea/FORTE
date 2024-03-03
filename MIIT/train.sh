export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/xx/checkpoints/OTTER-MPT7B-Init" \
--mimicit_path="/xx/data/MED_instruction.json" \
--images_path="/xx/data/MED_instruction.json" \
--batch_size=1 \
--gradient_accumulation_steps=4 \
--num_epochs=3 \
--report_to_wandb \
--wandb_entity="your institution" \
--run_name=OTTER-LLaMA7B-MED_CLIP \
--wandb_project=OTTER-LLaMA7B-MED_CLIP \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \