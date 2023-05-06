# ----------------------------------------CCT----------------------------------------------
python train.py -c configs/datasets/cifar100.yml --unit_only 0 --model_type cct --model cct_7_3x2_32 ./data/cifar100/

# -----------------------------------Collaborate-CCT--------------------------------------------
python train.py -c configs/datasets/cifar100.yml --unit_only 0 --model_type colla --model cct_7_3x2_32 ./data/cifar100/

# --------------------------------------Lin-CCT--------------------------------------------
python train.py -c configs/datasets/cifar100.yml --unit_only 0 --model_type lin --k 64 --model cct_7_3x2_32 ./data/cifar100/
python train.py -c configs/datasets/cifar100.yml --unit_only 0 --model_type lin --k 128 --model cct_7_3x2_32 ./data/cifar100/
python train.py -c configs/datasets/cifar100.yml --unit_only 0 --model_type lin --k 256 --model cct_7_3x2_32 ./data/cifar100/

# -------------------------------------Pshared-CCT--------------------------------------------
python train.py -c configs/datasets/cifar100.yml --unit_only 0 --model_type pshared --model cct_7_3x2_32 ./data/cifar100/

# --------------------------------------CP-CCT--------------------------------------------
python train.py -c configs/datasets/cifar100.yml --unit_only 0 --model_type cp --cp_rank 64 --theta 0. --is_decomposed 1 --model cct_7_3x2_32 ./data/cifar100/

# ratio = 0.25
python train.py -c configs/datasets/cifar100.yml --unit_only 0 --model_type cp --cp_rank 50 --theta 0. --is_decomposed 1 --model cct_7_3x2_32 ./data/cifar100/
# ratio = 0.5
python train.py -c configs/datasets/cifar100.yml --unit_only 0 --model_type cp --cp_rank 100 --theta 0. --is_decomposed 1 --model cct_7_3x2_32 ./data/cifar100/
# ratio = 0.75
python train.py -c configs/datasets/cifar100.yml --unit_only 0 --model_type cp --cp_rank 150 --theta 0. --is_decomposed 1 --model cct_7_3x2_32 ./data/cifar100/

# --------------------------------------BTD-CCT--------------------------------------------
python train.py -c configs/datasets/cifar100.yml --unit_only 0 --model_type btd --core_num 1 --model cct_7_3x2_32 ./data/cifar100/

python train.py -c configs/datasets/cifar100.yml --unit_only 0 --model_type btd --core_num 2 --model cct_7_3x2_32 ./data/cifar100/