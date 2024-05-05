
# The original S3T-Model
python Trans.py

# Lin-S3T
python linformer.py --k 64 --is_train 1
python linformer.py --k 128 --is_train 1
python linformer.py --k 256 --is_train 1

# Colla-S3T
python colla_attention.py

# paramters sharing
python pshared_Trans.py

# BTD
python btd_Trans.py --num_heads 1 --d_head 2 --is_train 1
python btd_Trans.py --num_heads 2 --d_head 2 --is_train 1

