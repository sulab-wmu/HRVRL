

<h1>HRVRL： An Efficient and Interpretable Foundation Model for Retinal Image Analysis in Disease Diagnosis</h1>

- <h3> Framework </h3>

<div style="text-align: center">
  <img src='fig/F1.png'/>
</div>

- <h3> Environment Requirements</h3>

```shell
git clone https://github.com/sulab-wmu/HRVRL.git
cd HRVRL
conda create -n hrvrl python=3.10 -y
conda activate hrvrl
pip install poetry 
poetry install
pip install tensorflow==2.9.1
pip install torch==1.13.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 --force
```

> <span style='color:red'>if you encounter the following problem</span>:
> AttributeError: module 'tensorflow' has no attribute 'GraphKeys'
> `vim /home/xxx/.conda/envs/xxx/lib/python3.10/site-packages/tensorlayer/layers.py`
> `import tensorflow as tf ---> import tensorflow.compat.v1 as tf`

- ### Finetune on your data

  Download the [pretrain weight](https://drive.google.com/drive/folders/1Hzxv36qyyqTgyE0jGICJ-1hlbg0DnSFh?usp=drive_link)  and run

  `cd finetune `

  ` sh run.sh or  CUDA_VISIBLE_DEVICES=0  python -m torch.distributed.launch --nproc_per_node=1 --master_port=xxx main_finetune_rip.py     --batch_size 16     --world_size 1     --epochs 100     --blr 8e-3 --layer_decay 0.9     --weight_decay 0.0005      --nb_classes [your class number]     --data_path [your dataset]    --task [your output dir]  --cutmix 0.1   --warmup_epochs 10   --drop_path 0.8 --finetune G_pretrain.pth --min_lr 1e-6  --input_size 512 > [your log file name] 2>&1 &` 

- ### Reproduction

  Download the [pretrain weight](https://drive.google.com/drive/folders/1Hzxv36qyyqTgyE0jGICJ-1hlbg0DnSFh?usp=drive_link) and [reproduction weight](https://drive.google.com/drive/folders/1U9kxh7_d-dXtFw6b8Gkpl4ZXd0bMgtbo?usp=drive_link) and put the target dir:

  - run `hrvrl_preprocess_augmentation.ipynb ` to get the demo and results of multi-level augmentation.

  - run `hrvrl_pretrain_av_visulization.ipynb ` to get the demo and results of pretrained feature visulization.

  - run `hrvrl_task_1_2_fig_plot_finetune.ipynb ` to get the result of Figure2.

  - run `hrvrl_explain_example.ipynb ` to get the demo and results of hierarchical explainable method.

- ### Acknowledge

  [RIP-AV](https://github.com/weidai00/RIP-AV)  and [RETFound](https://github.com/rmaphoh/RETFound_MAE) 

