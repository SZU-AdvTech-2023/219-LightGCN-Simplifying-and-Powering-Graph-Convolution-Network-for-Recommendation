## 数据集

Gowalla, Yelp2018 和 LastFM.   see more in `dataloader.py`

## An example to run a 3-layer LightGCN
运行neglightgcn模型在数据集gowalla上的代码为
` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20]" --recdim=64`



