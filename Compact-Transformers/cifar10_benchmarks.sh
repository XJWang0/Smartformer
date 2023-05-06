# ----------------------------------------CCT----------------------------------------------
python train.py -c configs/datasets/cifar10.yml --unit_only 0 --model_type cct --model cct_7_3x2_32 ./data/cifar10/

# -----------------------------------Collaborate-CCT--------------------------------------------
python train.py -c configs/datasets/cifar10.yml --unit_only 0 --model_type colla --model cct_7_3x2_32 ./data/cifar10/

# --------------------------------------Lin-CCT--------------------------------------------
python train.py -c configs/datasets/cifar10.yml --unit_only 0 --model_type lin --k 64 --model cct_7_3x2_32 ./data/cifar10/
python train.py -c configs/datasets/cifar10.yml --unit_only 0 --model_type lin --k 128 --model cct_7_3x2_32 ./data/cifar10/
python train.py -c configs/datasets/cifar10.yml --unit_only 0 --model_type lin --k 256 --model cct_7_3x2_32 ./data/cifar10/

# -------------------------------------Pshared-CCT--------------------------------------------
python train.py -c configs/datasets/cifar10.yml --unit_only 0 --model_type pshared --model cct_7_3x2_32 ./data/cifar10/

# --------------------------------------CP-CCT--------------------------------------------
python train.py -c configs/datasets/cifar10.yml --unit_only 0 --model_type cp --cp_rank 64 --theta 0. --is_decomposed 1 --model cct_7_3x2_32 ./data/cifar10/


# --------------------------------------BTD-CCT--------------------------------------------
python train.py -c configs/datasets/cifar10.yml --unit_only 0 --model_type btd --core_num 1 --model cct_7_3x2_32 ./data/cifar10/

python train.py -c configs/datasets/cifar10.yml --unit_only 0 --model_type btd --core_num 2 --model cct_7_3x2_32 ./data/cifar10/