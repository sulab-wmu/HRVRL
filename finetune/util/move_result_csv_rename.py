import os
import glob
import re
import shutil
import torch
target_name = 'binary_all_result_metric3'
target_path = r'/data1/sjz.wy/code/RETFound_MAE-main/finetune_HF'

all_files = glob.glob(os.path.join(target_path, '**', '*.pth'), recursive=True)
#pattern = re.compile(r'^.*/finetune_(.*)/.*img_(.*)/(a.*)_rip_512/a.*(train|test).*\.csv$')
#pattern = re.compile(r'^.*/finetune_(.*)/.*img_(.*)/(a.*)_rip_512/.*auc\.pth$')
#pattern = re.compile(r'^.*/finetune_(.*)/.*img_(.*)/(a.*)_rip_512/_metrics_test.*\.csv$')
#pattern = re.compile(r'^.*/finetune_(.*)/img_(.*)/weight_rip_512/_metrics_test.*\.csv$')

pattern = re.compile(r'^.*/finetune_(.*)/img_(.*)/rip_512_process3/.*auc\.pth$')

matched_files = [file for file in all_files if pattern.match(file)]
os.makedirs(os.path.join(target_path,target_name),exist_ok=True)
# 打印找到的文件列表
# for file in matched_files:
#     match = pattern.search(file)
#     dataset_name_short = match.group(1)
#     img_type = match.group(2)
#     dataset_binary = match.group(3)
#     train_test = match.group(4)
#     dst = os.path.join(target_path,target_name,f'{dataset_name_short}_{img_type}_{dataset_binary}_{train_test}_prediction_list.csv')
#     shutil.copy(file,dst)
#     print(dst)
    
for file in matched_files:
    #print(file)
    match = pattern.search(file)
    dataset_name_short = match.group(1)
    img_type = match.group(2)
    #dataset_binary = match.group(3)
    pt = torch.load(file)
    pt_model = pt['model']
    #dst = os.path.join(target_path,target_name,f'{dataset_name_short}_{img_type}_{dataset_binary}_checkpoint.pth')
    dst = os.path.join(target_path,target_name,f'{dataset_name_short}_{img_type}_checkpoint.pth')
    #shutil.copy(file,dst)
    torch.save(pt_model,dst)
    print(dst)
    
# for file in matched_files:
#     match = pattern.search(file)
#     dataset_name_short = match.group(1)
#     img_type = match.group(2)
#     #dataset_binary = match.group(3)
    
#     #dst = os.path.join(target_path,target_name,f'{dataset_name_short}_{img_type}_{dataset_binary}_test_metric.csv')
#     dst = os.path.join(target_path,target_name,f'{dataset_name_short}_{img_type}_cold_test_prediction_list.csv')
#     shutil.copy(file,dst)
#     print(dst)

