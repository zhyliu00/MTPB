# Multi-scale Traffic Pattern Bank for Cross-city Few-shot Traffic Forecasting

## Data

The data is in https://drive.google.com/drive/folders/1UrKTgR27YmP9PjJ-FWv4SCDH3zUxtc5R?usp=share_link.
Please download it and save them in `./data`

The pre-trained model is in https://drive.google.com/file/d/1URbZrV4UF6yxq7pP0PhEKJ_C8Ft9cO1b/view?usp=sharing.
Please download it and save them in `./save`

The pretrain model is in 
## Environment
The code is implemented in pytorch 1.10.0, CUDA version 11.3, python 3.7.0.

```bash
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

## How to train
To train the model in all setting, please run following command:
```bash
bash train.sh
```
or run the experiment on specific dataset with `64`'threads and gpu `0` (PEMS-BAY as an example):
```bash
bash train_one.sh pems-bay 64 0
```

The result is in ./out/\${data_list}/ , where \${data_list} is the source data. For example, if you pre-train in `PEMS-BAY` then the \${data_list} is `chengdu_shenzhen_metr`.

## Pre-trained stuff

The pre-trained patch encoder and traffic pattern bank is contained in this repository.
The pre-trained patch encoder is in `./save/pretrain_model` and the traffic pattern bank is in `./pattern`.

You can also pre-train and generate traffic pattern bank on your own by:
```bash
# Pre-train
bash pretrain.sh ${test_dataset} ${threads} ${gpu}
wait
bash patch_devide.sh ${test_dataset} ${threads} ${gpu}
wait
bash pattern_clustering.sh ${test_dataset} ${threads} ${gpu}
```

`${test_dataset}` is the dataset you want to build target data on. If you want to build target data on `Shenzhen` then the `${test_dataset}` is `shenzhen`.
