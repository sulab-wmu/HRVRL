# pretrain
#CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=4  --master_port=48798 main_sim.py   --batch_size 32 --lr 4e-3 --update_freq 1  --epochs 100  --data_path /data1/sjz.wy/code/RIP-AV-main/RIP/dataset_all_b/  --input_size 256 --output_dir ./output_all_b_o_sim_cls2_512 --log_dir ./log/dataset_all_b_o_sim_cls2_512  > log_b_o_sim_cls2_512.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1  --master_port=48798 main_sim.py   --batch_size 128 --lr 4e-3 --update_freq 1  --epochs 100  --data_path /data1/sjz.wy/code/RIP-AV-main/RIP/dataset_all_b_v2/  --input_size 256 --output_dir ./output_all_b_o_sim_cls2_v2 --log_dir ./log/dataset_all_b_o_sim_cls2_v2  > log_b_o_sim_cls2_v2.txt 2>&1 &

#finetune
# finetune APTOS2019
#CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1  --master_port=48791 main_finetune_simclr.py   --batch_size 16 --lr 5e-4  --nb_classes 5 --update_freq 1 --warmup_epochs 10 --epochs 100 --layer_decay 0.9  --smoothing 0.1 --weight_decay 5e-5 --input_size 512 --finetune ./output_all_b_o_sim/checkpoint-best.pth --data_path /data1/sjz.wy/dataset/APTOS2019/  --output_dir ./retfound_compare_log_process/retfound_APTOS2019_sim/ --log_dir ./retfound_compare_log_process/retfound_APTOS2019_sim_log  > ./retfound_compare_log_process/log_retfound_APTOS2019_sim.txt 2>&1 &




#python -m torch.distributed.launch --nproc_per_node=1 main.py   --batch_size 128 --lr 4e-3 --update_freq 1 --model_ema true --model_ema_eval true --data_path /data1/sjz.wy/code/RIP-AV-main/RIP/dataset_STU_v2/ --resume ./output/checkpoint-299.pth  c --start_epoch 300 --epochs 500 

#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4  --master_port=48798  main.py   --batch_size 128 --lr 4e-3 --update_freq 1 --model_ema true --model_ema_eval true --data_path /data1/sjz.wy/code/RIP-AV-main/RIP/dataset_all_v2/ --output_dir ./output_all 

#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4  --master_port=48798  main.py   --batch_size 128 --lr 4e-3 --warmup_epochs 10 --start_epoch 300 --epochs 600 --update_freq 1 --model_ema true --model_ema_eval true --data_path /data1/sjz.wy/code/RIP-AV-main/RIP/dataset_all_v3/  --resume /data1/sjz.wy/code/RIP-AV-main/PRE/output_all/checkpoint-299.pth --output_dir ./output_all_bg 
#CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3  --master_port=48798  main.py   --batch_size 128 --lr 4e-3 --warmup_epochs 0 --start_epoch 300 --epochs 600 --update_freq 1 --model_ema true --model_ema_eval true --data_path /data1/sjz.wy/code/RIP-AV-main/RIP/dataset_all_v3/  --output_dir ./output_all_bg 

#python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py   --batch_size 16 --lr 5e-4  --data_path /home/sjz.wy/tmp/DR_dataset/ --finetune ./output_all/checkpoint-best.pth  --warmup_epochs 10 --start_epoch 0 --epochs 100  --weight_decay 1e-8 --output_dir ./output_dr --log_dir ./log/ukbb_dr 


# finetune
#python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py   --batch_size 16 --lr 5e-4  --data_path /home/sjz.wy/tmp/DR_dataset/ --finetune ./output_all/checkpoint-best-ema.pth  --warmup_epochs 10 --start_epoch 0 --epochs 150 --update_freq 2  --layer_decay 0.9  --smoothing 0.1  --weight_decay 5e-4 --output_dir ./output_dr_metric/ --log_dir ./log/ukbb_dr/


#python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py   --batch_size 16 --lr 5e-5  --data_path /home/sjz.wy/tmp/DR_dataset/ --finetune ./output_dr_metric/checkpoint-best.pth  --warmup_epochs 0 --start_epoch 0 --epochs 150 --update_freq 2  --layer_decay 0.9  --smoothing 0.1  --weight_decay 5e-4 --output_dir ./output_dr_metric_250/ --log_dir ./log/ukbb_dr/
#feature
#python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py   --batch_size 16 --lr 5e-3  --data_path /home/sjz.wy/tmp/DR_dataset/ --finetune ./output_all/checkpoint-best-ema.pth  --warmup_epochs 0 --start_epoch 0 --epochs 150 --update_freq 2  --layer_decay 0.9  --smoothing 0.0  --weight_decay 5e-5 --output_dir ./output_dr_metric_wd5e_5_ref/ --log_dir ./log/ukbb_dr/ 
