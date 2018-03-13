# DeepLab

DeepLab implementation in pytorch.
DeepLabv3の方が精度は高い一方で推論時間は若干遅くなります．

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
                [--save_steps SAVE_STEPS] [--weight_decay WEIGHT_DECAY]

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
```

## Infer

```bash
usage: infer.py [-h] [--model MODEL] [--data-dir DATA_DIR]
                [--data-list DATA_LIST] [--ignore-label IGNORE_LABEL]
                [--num-classes NUM_CLASSES] [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL
  --data-dir DATA_DIR
  --data-list DATA_LIST
  --ignore-label IGNORE_LABEL
  --num-classes NUM_CLASSES
  --gpu GPU             choose gpu device.
```
