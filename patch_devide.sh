
test_dataset=$1
threads=$2
gpu=$3


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

OMP_NUM_THREADS=${threads} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u patch_devide.py --test_dataset ${test_dataset} --data_list ${data_list} --config_filename ./configs/config.yaml > ./out/${data_list}/patch_devide_${test_dataset}.out 2>&1 &