
test_dataset=$1
threads=$2
gpu=$3

if [ ${test_dataset} == 'chengdu_m' ];
then
data_list='shenzhen_pems_metr'
fi

if [ ${test_dataset} == "shenzhen" ];
then
echo 'chengdu_pems_metr'
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
mkdir ./out/pretrain/

echo "OMP_NUM_THREADS=${threads} CUDA_VISIBLE_DEVICES=${gpu}  nohup python -u ./pretrain.py --test_dataset ${test_dataset} --data_list $data_list --config_filename ./configs/config.yaml > ./out/pretrain/${data_list}.out 2>&1 &"
OMP_NUM_THREADS=${threads} CUDA_VISIBLE_DEVICES=${gpu}  nohup python -u ./pretrain.py --test_dataset ${test_dataset} --data_list $data_list --config_filename ./configs/config.yaml > ./out/pretrain/${data_list}.out 2>&1 &