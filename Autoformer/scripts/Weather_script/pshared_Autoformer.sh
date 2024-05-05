export CUDA_VISIBLE_DEVICES=0


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --checkpoints ./checkpoints/pshared/ \
  --model_id weather_96_96 \
  --model param_shared \
  --data custom \
  --features M \
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
  --train_epochs 2 > log/pshared/pshared_Autoformer_weather_M_96.txt

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --checkpoints ./checkpoints/pshared/ \
  --model_id weather_96_192 \
  --model param_shared \
  --data custom \
  --features M \
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
  --itr 1 > log/pshared/pshared_Autoformer_weather_M_192.txt

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --checkpoints ./checkpoints/pshared/ \
  --model_id weather_96_336 \
  --model param_shared \
  --data custom \
  --features M \
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
  --itr 1 > log/pshared/pshared_Autoformer_weather_M_336.txt

<< Comment
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --checkpoints ./checkpoints/pshared/ \
  --model_id weather_96_720 \
  --model param_shared \
  --data custom \
  --features M \
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
  --itr 1 > log/pshared/pshared_Autoformer_weather_M_720.txt
Comment