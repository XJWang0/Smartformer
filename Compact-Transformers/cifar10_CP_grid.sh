
# Cifar10  CP-CCT grid
# ratio = 0.25
python train.py -c configs/datasets/cifar10.yml --unit_only 0 --model_type cp --cp_rank 50 --theta 0. --is_decomposed 1 --model cct_7_3x2_32 ./data/cifar10/
# ratio = 0.5
python train.py -c configs/datasets/cifar10.yml --unit_only 0 --model_type cp --cp_rank 100 --theta 0. --is_decomposed 1 --model cct_7_3x2_32 ./data/cifar10/
# ratio = 0.75
python train.py -c configs/datasets/cifar10.yml --unit_only 0 --model_type cp --cp_rank 150 --theta 0. --is_decomposed 1 --model cct_7_3x2_32 ./data/cifar10/
