
# ---------------------------------------SVD--------------------------------------------
python smart_train.py -c configs/datasets/cifar100.yml --unit_only 0 --model_type svd --theta 0. --cp_rank 128 --terminal_num 3 --cp_layers 7 --is_decomposed 1 --model cct_7_3x2_32 ./data/cifar100/


# -----------------------------------Smartformer--------------------------------------------
python smart_train.py -c configs/datasets/cifar100.yml --unit_only 0 --model_type smart --theta 0. --cp_rank 200 --terminal_num 3 --cp_layers 7 --is_decomposed 1 --model cct_7_3x2_32 ./data/cifar100/

