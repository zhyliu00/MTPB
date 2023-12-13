test_dataset=$1
threads=$2
gpu=$3

Ks=(10)

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

for K in ${Ks[@]}
    do
    OMP_NUM_THREADS=${threads} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u pattern_clustering.py --K ${K} --test_dataset ${test_dataset} --data_list ${data_list} > ./out/${data_list}/pattern_clustering_${test_dataset}_K${K}.out 2>&1 &
    wait
    done