# kosen_research

## ファイル構成
config/model: train.pyに引き渡す時のモデルの情報
config/data: データセット
src/train.py: trainとvalで学習用
src/model/*: モデルの定義

## 実行方法
```
python src/train model=<model_name>
```
