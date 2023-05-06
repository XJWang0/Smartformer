python -u main_informer.py \
      --checkpoints ./checkpoints/CP_Informer/ \
      --start_rank 90 \
      --model CP_informer \
      --data ETTh1 \
      --features S \
      --seq_len 720 \
      --label_len 168 \
      --pred_len 24 \
      --e_layers 2 \
      --d_layers 1 \
      --attn prob \
      --des 'Exp' \
      --itr 5
#