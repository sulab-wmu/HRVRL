# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46591 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/ukbb_av_for_train/img_nav/cerebrovasular    --task ./finetune_CE/img_nav/rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10  --eval  --resume ./finetune_CE/img_nav/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512   &

# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=43592 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/ukbb_av_for_train/img_nav/diabetes_part    --task ./finetune_DE/img_nav/rip_5122/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval --resume ./finetune_DE/img_nav/rip_5122/checkpoint-best.pth  --min_lr 5e-6 --drop_path 0.1 --input_size 512  &
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46593 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/ukbb_av_for_train/img_nav/hy_part    --task ./finetune_HY/img_nav/rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10  --eval  --resume ./finetune_HY/img_nav/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512 &
# CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46594 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 5     --data_path /data1/sjz.wy/dataset/ukbb_av_for_train/img_nav/eye_disease    --task ./finetune_ED/img_nav/rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval  --resume ./finetune_ED/img_nav/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512 &
# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46595 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/ukbb_av_for_train/img_nav/parkinson    --task ./finetune_PD/img_nav/rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval  --resume ./finetune_PD/img_nav/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512 &
# CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46296 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/ukbb_av_for_train/img_nav/myocardial_infarction    --task ./finetune_MY/img_nav/rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval  --resume ./finetune_MY/img_nav/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512  &

# CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46527 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/ukbb_av_for_train/img_nav/heart_failure    --task ./finetune_HF/img_nav/rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval  --resume ./finetune_HF/img_nav/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512 &


# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46591 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 5     --data_path /data1/sjz.wy/dataset/APT_av_for_train/img_a    --task ./finetune_APT/img_a/rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10  --eval  --resume ./finetune_APT/img_a/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512   &
# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=43592 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 5     --data_path /data1/sjz.wy/dataset/APT_av_for_train/img_v    --task ./finetune_APT/img_v/rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval --resume ./finetune_APT/img_v/rip_512/checkpoint-best.pth  --min_lr 5e-6 --drop_path 0.1 --input_size 512  &
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46593 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 5     --data_path /data1/sjz.wy/dataset/APT_av_for_train/img_av    --task ./finetune_APT/img_av/rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10  --eval  --resume ./finetune_APT/img_av/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512 &
# CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46594 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 5     --data_path /data1/sjz.wy/dataset/APT_av_for_train/img_na    --task ./finetune_APT/img_na/rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval  --resume ./finetune_APT/img_na/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512 &
# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46595 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 5     --data_path /data1/sjz.wy/dataset/APT_av_for_train/img_nv    --task ./finetune_APT/img_nv/rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval  --resume ./finetune_APT/img_nv/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512 &
# CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46296 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 5     --data_path /data1/sjz.wy/dataset/APT_av_for_train/img_nav    --task ./finetune_APT/img_nav/rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval  --resume ./finetune_APT/img_nav/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512  &




# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46591 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/ukbb_av_for_train/img_na/cerebrovasular    --task ./finetune_CE/img_na/rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10  --eval  --resume ./finetune_CE/img_na/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512   &

# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=43592 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/ukbb_av_for_train/img_na/diabetes_part    --task ./finetune_DE/img_na/rip_5122/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval --resume ./finetune_DE/img_na/rip_5122/checkpoint-best.pth  --min_lr 5e-6 --drop_path 0.1 --input_size 512  &
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46593 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/ukbb_av_for_train/img_na/hy_part    --task ./finetune_HY/img_na/rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10  --eval  --resume ./finetune_HY/img_na/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512 &
# CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46594 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 5     --data_path /data1/sjz.wy/dataset/ukbb_av_for_train/img_na/eye_disease    --task ./finetune_ED/img_na/rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval  --resume ./finetune_ED/img_na/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512 &
# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46595 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/ukbb_av_for_train/img_na/parkinson    --task ./finetune_PD/img_na/rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval  --resume ./finetune_PD/img_na/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512 &
# CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46296 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/ukbb_av_for_train/img_na/myocardial_infarction    --task ./finetune_MY/img_na/rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval  --resume ./finetune_MY/img_na/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512  &

# CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46527 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/ukbb_av_for_train/img_na/heart_failure    --task ./finetune_HF/img_na/rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval  --resume ./finetune_HF/img_na/rip_512/checkpoint-best.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512 &



# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46591 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 5     --data_path /data1/sjz.wy/dataset/MESSIDOR2_av_for_train/img_a    --task ./finetune_MESS/img_a/weight_rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10  --eval  --resume ./finetune_MESS/img_a/weight_rip_512/checkpoint-best_auc.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512   &
# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=43592 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 5     --data_path /data1/sjz.wy/dataset/MESSIDOR2_av_for_train/img_v    --task ./finetune_MESS/img_v/weight_rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval --resume ./finetune_MESS/img_v/weight_rip_512/checkpoint-best_auc.pth  --min_lr 5e-6 --drop_path 0.1 --input_size 512  &
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46593 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 5     --data_path /data1/sjz.wy/dataset/MESSIDOR2_av_for_train/img_av    --task ./finetune_MESS/img_av/weight_rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10  --eval  --resume ./finetune_MESS/img_av/weight_rip_512/checkpoint-best_auc.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512 &
# CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46594 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 5     --data_path /data1/sjz.wy/dataset/MESSIDOR2_av_for_train/img_na    --task ./finetune_MESS/img_na/weight_rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval  --resume ./finetune_MESS/img_na/weight_rip_512/checkpoint-best_auc.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512 &
# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46595 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 5     --data_path /data1/sjz.wy/dataset/MESSIDOR2_av_for_train/img_nv    --task ./finetune_MESS/img_nv/weight_rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval  --resume ./finetune_MESS/img_nv/weight_rip_512/checkpoint-best_auc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512 &
# CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46296 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 5     --data_path /data1/sjz.wy/dataset/MESSIDOR2_av_for_train/img_nav    --task ./finetune_MESS/img_nav/weight_rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval  --resume ./finetune_MESS/img_nav/weight_rip_512/checkpoint-best_auc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512  &



# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46536 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 50     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005  --drop_path 0.8      --nb_classes 2     --data_path /data1/sjz.wy/dataset/MESSIDOR2    --task ./finetune_MESS/img_full/ab_rip_512/  --mixup 0.0 --cutmix 0.0  --warmup_epochs 10 --min_lr 1e-6  --eval  --resume  ./finetune_MESS/img_full/ab_rip_512/checkpoint-best_auc.pth --input_size 512 &
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46591 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/MESSIDOR2_av_for_train/img_a    --task ./finetune_MESS/img_a/ab_rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10  --eval  --resume ./finetune_MESS/img_a/ab_rip_512/checkpoint-best_auc.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512   &
# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=43592 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/MESSIDOR2_av_for_train/img_v    --task ./finetune_MESS/img_v/ab_rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval --resume ./finetune_MESS/img_v/ab_rip_512/checkpoint-best_auc.pth  --min_lr 5e-6 --drop_path 0.1 --input_size 512  &
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46593 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/MESSIDOR2_av_for_train/img_av    --task ./finetune_MESS/img_av/ab_rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10  --eval  --resume ./finetune_MESS/img_av/ab_rip_512/checkpoint-best_auc.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512 &
# CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46594 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/MESSIDOR2_av_for_train/img_na    --task ./finetune_MESS/img_na/ab_rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval  --resume ./finetune_MESS/img_na/ab_rip_512/checkpoint-best_auc.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512 &
# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46595 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/MESSIDOR2_av_for_train/img_nv    --task ./finetune_MESS/img_nv/ab_rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval  --resume ./finetune_MESS/img_nv/ab_rip_512/checkpoint-best_auc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512 &
# CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46296 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/MESSIDOR2_av_for_train/img_nav    --task ./finetune_MESS/img_nav/ab_rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval  --resume ./finetune_MESS/img_nav/ab_rip_512/checkpoint-best_auc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512  &


# CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46299 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 50     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/hy_re    --task ./finetune_HR/img/rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval  --resume ./finetune_HR/img/rip_512/checkpoint-best_acc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512  &
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46591 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 50     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/hy_re_av_for_train/img_a    --task ./finetune_HR/img_a/rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10  --eval  --resume ./finetune_HR/img_a/rip_512/checkpoint-best_acc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512   &
# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=43592 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 50     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2    --data_path /data1/sjz.wy/dataset/hy_re_av_for_train/img_v    --task ./finetune_HR/img_v/rip_512/    --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval --resume ./finetune_HR/img_v/rip_512/checkpoint-best_acc.pth  --min_lr 5e-6 --drop_path 0.1 --input_size 512  &
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46593 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 50     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/hy_re_av_for_train/img_av    --task ./finetune_HR/img_av/rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10  --eval  --resume ./finetune_HR/img_av/rip_512/checkpoint-best_acc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512 &
# CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46594 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 50     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/hy_re_av_for_train/img_na    --task ./finetune_HR/img_na/rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval  --resume ./finetune_HR/img_na/rip_512/checkpoint-best_acc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512 &
# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46595 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 50     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/hy_re_av_for_train/img_nv    --task ./finetune_HR/img_nv/rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval  --resume ./finetune_HR/img_nv/rip_512/checkpoint-best_acc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512 &
# CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46296 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 50     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/hy_re_av_for_train/img_nav    --task ./finetune_HR/img_nav/rip_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval  --resume ./finetune_HR/img_nav/rip_512/checkpoint-best_acc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512  &


# CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46299 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 50     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/hy_re    --task ./finetune_HR/img/rip_cold_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval  --resume ./finetune_HR/img/rip_cold_512/checkpoint-best_acc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512  &
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46591 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 50     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/hy_re_av_for_train/img_a    --task ./finetune_HR/img_a/rip_cold_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10  --eval  --resume ./finetune_HR/img_a/rip_cold_512/checkpoint-best_acc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512   &
# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=43592 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 50     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2    --data_path /data1/sjz.wy/dataset/hy_re_av_for_train/img_v    --task ./finetune_HR/img_v/rip_cold_512/    --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval --resume ./finetune_HR/img_v/rip_cold_512/checkpoint-best_acc.pth  --min_lr 5e-6 --drop_path 0.1 --input_size 512  &
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46593 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 50     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/hy_re_av_for_train/img_av    --task ./finetune_HR/img_av/rip_cold_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10  --eval  --resume ./finetune_HR/img_av/rip_cold_512/checkpoint-best_acc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512 &
# CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46594 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 50     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/hy_re_av_for_train/img_na    --task ./finetune_HR/img_na/rip_cold_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval  --resume ./finetune_HR/img_na/rip_cold_512/checkpoint-best_acc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512 &
# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46595 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 50     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/hy_re_av_for_train/img_nv    --task ./finetune_HR/img_nv/rip_cold_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval  --resume ./finetune_HR/img_nv/rip_cold_512/checkpoint-best_acc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512 &
# CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46296 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 50     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/hy_re_av_for_train/img_nav    --task ./finetune_HR/img_nav/rip_cold_512/   --mixup 0.0 --cutmix 0.0  --warmup_epochs 10   --eval  --resume ./finetune_HR/img_nav/rip_cold_512/checkpoint-best_acc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512  &


#CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46296 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 4     --data_path /data1/sjz.wy/dataset/brset_multi_disease    --task ./finetune_BR/rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval  --resume ./finetune_BR/rip_512/checkpoint-best_auc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512  &
# CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46246 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 4     --data_path /data1/sjz.wy/dataset/brset_multi_disease    --task ./finetune_BR/rip_512_cold/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval  --resume ./finetune_BR/rip_512_cold/checkpoint-best_auc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512  &
#CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46246 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/ukbb_for_train/heart_failure    --task ./finetune_HF/rip_512/   --mixup 0.1 --cutmix 0.1  --warmup_epochs 10   --eval  --resume ./finetune_HF/rip_512/checkpoint-best_auc.pth --min_lr 1e-6 --drop_path 0.1 --input_size 512  &




CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46246 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/ukbb_for_train/myocardial_infarction_process    --task ./finetune_MY/img_full/rip_512_process/   --mixup 0.0 --cutmix 0.1  --warmup_epochs 10   --eval  --resume ./finetune_MY/img_full/rip_512_process/checkpoint-best_auc.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512  &
# CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=46246 main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes 2     --data_path /data1/sjz.wy/dataset/ukbb_av_for_train/img_a/myocardial_infarction_process   --task ./finetune_MY/img_full/rip_512_process/   --mixup 0.0 --cutmix 0.1  --warmup_epochs 10   --eval  --resume ./finetune_MY/img_full/rip_512_process/checkpoint-best_auc.pth --min_lr 1e-6 --drop_path 0.8 --input_size 512  &
