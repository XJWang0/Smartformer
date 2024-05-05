export CUDA_VISIBLE_DEVICES=0,1

python -u run.py \
  --is_training 1 \
  --rank 90 \
  --checkpoints ./checkpoints/CP_Autoformer/ \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model CP_Autoformer \
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
  --train_epochs 2 > log/CP_Autoformer/CP_Autoformer_weather_M_96_R_90.txt

python -u run.py \
  --is_training 1 \
  --rank 90 \
  --checkpoints ./checkpoints/CP_Autoformer/ \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model CP_Autoformer \
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
  --itr 1 > log/CP_Autoformer/CP_Autoformer_weather_M_192_R_90.txt

python -u run.py \
  --is_training 1 \
  --rank 90 \
  --checkpoints ./checkpoints/CP_Autoformer/ \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model CP_Autoformer \
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
  --itr 1 > log/CP_Autoformer/CP_Autoformer_weather_M_336_R_90.txt
  
  
