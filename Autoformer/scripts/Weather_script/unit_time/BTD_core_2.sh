export CUDA_VISIBLE_DEVICES=0

:<<!
这是一整段注释
!

python -u run.py \
  --is_training 1 \
  --train 0 \
  --test 0 \
  --concat1 9216 \
  --concat2 20736 \
  --checkpoints ./checkpoints/BTD/ \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model BTDformer \
  --data custom \
  --features M \
  --n_heads 2 \
  --d_head 64\
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 2 > log/BTD/heads_2_dv_64/unit_processing_96.txt

python -u run.py \
  --is_training 1 \
  --train 0 \
  --test 0 \
  --concat1 9216 \
  --concat2 57600 \
  --checkpoints ./checkpoints/BTD/ \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model BTDformer \
  --data custom \
  --features M \
  --n_heads 2 \
  --d_head 64\
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 > log/BTD/heads_2_dv_64/unit_processing_192.txt

python -u run.py \
  --is_training 1 \
  --train 0 \
  --test 0 \
  --concat1 9216 \
  --concat2 147456 \
  --checkpoints ./checkpoints/BTD/ \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model BTDformer \
  --data custom \
  --features M \
  --n_heads 2 \
  --d_head 64\
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 > log/BTD/heads_2_dv_64/unit_processing_336.txt

python -u run.py \
  --is_training 1 \
  --train 0 \
  --test 0 \
  --concat1 9216 \
  --concat2 589824 \
  --checkpoints ./checkpoints/BTD/ \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model BTDformer \
  --data custom \
  --features M \
  --n_heads 2 \
  --d_head 64\
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 > log/BTD/heads_2_dv_64/unit_processing_720.txt