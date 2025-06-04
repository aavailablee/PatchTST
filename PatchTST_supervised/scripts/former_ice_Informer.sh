# ALL scripts in this file come from Autoformer
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting10" ]; then
    mkdir ./logs/LongForecasting10
fi

random_seed=2025
for model_name in Informer
do 
for pred_len in 24
do
  python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id ice_48_$pred_len \
    --model $model_name \
    --data ice \
    --features MS \
    --seq_len 48 \
    --label_len 0 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 4 \
    --dec_in 4 \
    --c_out 4 \
    --des 'Exp' \
    --train_epoch 10 \
    --patience 10 \
    --itr 1 >logs/LongForecasting10MS/$model_name'_ice_48_0_'$pred_len.log
done
done