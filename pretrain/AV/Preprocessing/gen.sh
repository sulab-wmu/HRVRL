nohup python generate_patch_selection_for_RIP_all_b.py --dataset_path ../data/STU --train_or_test training --patch_size 96 128 256 384 --output ../../RIP/dataset_all_b --test_ind 30  >> STU_train.txt 2>&1  &
nohup python generate_patch_selection_for_RIP_all_b.py --dataset_path ../data/STU --train_or_test test --patch_size 96 128 256 384 --output ../../RIP/dataset_all_b --test_ind 30 >>STU_test.txt 2>&1  &

nohup python generate_patch_selection_for_RIP_all_b.py --dataset_path ../data/AV_DRIVE --train_or_test training --patch_size 96 128 256 --output ../../RIP/dataset_all_b --test_ind 15  >> DRIVE_train.txt 2>&1  &
nohup python generate_patch_selection_for_RIP_all_b.py --dataset_path ../data/AV_DRIVE --train_or_test test --patch_size 96 128 256  --output ../../RIP/dataset_all_b --test_ind 15 >>DRIVE_test.txt 2>&1  &

nohup python generate_patch_selection_for_RIP_all_b.py --dataset_path ../data/hrf --train_or_test training --patch_size 96 128 256 384 --output ../../RIP/dataset_all_b --test_ind 20  >> hrf_train.txt 2>&1  &
nohup python generate_patch_selection_for_RIP_all_b.py --dataset_path ../data/hrf --train_or_test test --patch_size  96 128 256 384  --output ../../RIP/dataset_all_b --test_ind 20 >>hrf_test.txt 2>&1  &

nohup python generate_patch_selection_for_RIP_all_b.py --dataset_path ../data/LES_AV --train_or_test training --patch_size 96 128 256 384 --output ../../RIP/dataset_all_b --test_ind 8  >> LES_train.txt 2>&1  &
nohup python generate_patch_selection_for_RIP_all_b.py --dataset_path ../data/LES_AV --train_or_test test --patch_size 96 128 256 384 --output ../../RIP/dataset_all_b --test_ind 8 >>LES_test.txt 2>&1  &

nohup python generate_patch_selection_for_RIP_all_b.py --dataset_path ../data/ukbb --train_or_test training --patch_size 96 128 256 384 --output ../../RIP/dataset_all_b --test_ind 40  >> ukbb_train.txt 2>&1  &
nohup python generate_patch_selection_for_RIP_all_b.py --dataset_path ../data/ukbb --train_or_test test --patch_size 96 128 256  384 --output ../../RIP/dataset_all_b --test_ind 40 >>ukbb_test.txt 2>&1  &

