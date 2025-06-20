if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./logs/LongForecasting10" ]; then
    mkdir ./logs/LongForecasting10
fi
seq_len=48
model_name=PatchTST

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=ice
data_name=ice

random_seed=2021
for pred_len in 24
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features MS \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 4 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 12\
      --stride 12\
      --des 'Exp' \
      --train_epochs 3\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 24 --learning_rate 0.0001 >logs/LongForecasting10MS/$model_name'_plst12'$model_id_name'_'$seq_len'_'$pred_len.log 
done


# pl16 stride8 0.018