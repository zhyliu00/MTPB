

test_dataset=$1
threads=$2
gpu=$3

adj_alpha=10
adj_k=20
train_epochs=50
finetune_epochs=300
update_step=2


if [ ${test_dataset} == 'chengdu_m' ];
then
data_list='shenzhen_pems_metr'
fi

if [ ${test_dataset} == "shenzhen" ];
then
data_list='chengdu_metr_pems'
fi

if [ ${test_dataset} == 'metr-la' ];
then
data_list='chengdu_shenzhen_pems'
fi

if [ ${test_dataset} == 'pems-bay' ];
then
data_list='chengdu_shenzhen_metr'
fi

mkdir ./out/
mkdir ./out/${data_list}/

OMP_NUM_THREADS=${threads} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u train.py  --adj_alpha ${adj_alpha} --adj_k $adj_k --update_step $update_step --finetune_epochs $finetune_epochs --train_epochs $train_epochs --data_list ${data_list} --test_dataset ${test_dataset} --config_file ./configs/config.yaml > ./out/${data_list}/train_${test_dataset}.out 2>&1 &

echo "OMP_NUM_THREADS=${threads} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u train.py  --adj_alpha ${adj_alpha} --adj_k $adj_k --update_step $update_step --finetune_epochs $finetune_epochs --train_epochs $train_epochs --data_list ${data_list} --test_dataset ${test_dataset} --config_file ./configs/config.yaml > ./out/${data_list}/train_${test_dataset}.out 2>&1 &"