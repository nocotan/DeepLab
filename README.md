# DeepLab

DeepLab implementation in pytorch.  
DeepLabv3の方が精度は高い一方で推論時間は若干遅くなります．

* v1: Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs([link](https://arxiv.org/abs/1412.7062))
* v2: DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs([link](https://arxiv.org/abs/1606.00915))
* v3: Rethinking Atrous Convolution for Semantic Image Segmentation([link](https://arxiv.org/abs/1706.05587))

## Usage

* datasets/train.txtに拡張子なしの画像ファイル名を列挙
* datasets/val.txtに拡張子なしのテスト用画像ファイル名を列挙
* datasets/imagesディレクトリにオリジナル画像を*.jpg形式で入れる
* datasets/labelsディレクトリにラベル画像を*.png形式で入れる

## Train

```bash
usage: train.py [-h] [--model MODEL] [--gpu GPU] [--batch_size BATCH_SIZE]
                [--data_dir DATA_DIR] [--data_list DATA_LIST]
                [--ignore_label IGNORE_LABEL] [--input_size INPUT_SIZE]
                [--lr LR] [--momentum MOMENTUM] [--num_classes NUM_CLASSES]
                [--num_steps NUM_STEPS] [--power POWER] [--random_mirror]
                [--random_scale] [--snapshot_dir SNAPSHOT_DIR]
                [--save_steps SAVE_STEPS] [--weight_decay WEIGHT_DECAY] [--v3]
                [--distributed]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL
  --gpu GPU
  --batch_size BATCH_SIZE
  --data_dir DATA_DIR
  --data_list DATA_LIST
  --ignore_label IGNORE_LABEL
  --input_size INPUT_SIZE
  --lr LR
  --momentum MOMENTUM
  --num_classes NUM_CLASSES
  --num_steps NUM_STEPS
  --power POWER
  --random_mirror
  --random_scale
  --snapshot_dir SNAPSHOT_DIR
  --save_steps SAVE_STEPS
  --weight_decay WEIGHT_DECAY
  --v3
  --distributed
```

## Infer

```bash
usage: infer.py [-h] [--model MODEL] [--data-dir DATA_DIR]
                [--data-list DATA_LIST] [--ignore-label IGNORE_LABEL]
                [--num-classes NUM_CLASSES] [--v3] [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL
  --data-dir DATA_DIR
  --data-list DATA_LIST
  --ignore-label IGNORE_LABEL
  --num-classes NUM_CLASSES
  --v3
  --gpu GPU             choose gpu device.
```
