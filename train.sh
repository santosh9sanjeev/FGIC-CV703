# Inception-V4 (Baseline)

python train.py --dataset "aircraft" --data_dir "/apps/local/shared/CV703/datasets/" --num-classes 100 --model "inception_v4" --pretrained -b 32 --epochs 800 --lr 0.025 --weight-decay 0.0005 --decay-epochs 10 --decay-rate 0.9 --mixup 0.2 --aa rand-m9-mstd0.5-inc1 --reprob 0.5 --remode pixel --output "./experiments/task1/self_attn_v2" -j 8 --amp --native-amp

python train.py --dataset "task2" --data_dir "/apps/local/shared/CV703/datasets/" --num-classes 296 --model "inception_v4" --pretrained -b 32 --epochs 800 --lr 0.025 --weight-decay 0.0005 --decay-epochs 10 --decay-rate 0.9 --mixup 0.2 --aa rand-m9-mstd0.5-inc1 --reprob 0.5 --remode pixel --output "./experiments/task2/self_attn_v2" -j 8 --amp --native-amp

python train.py --dataset "task3" --data_dir "/apps/local/shared/CV703/datasets/FoodX/food_dataset" --num-classes 251 --model "inception_v4" --pretrained -b 32 --epochs 800 --lr 0.025 --weight-decay 0.0005 --decay-epochs 10 --decay-rate 0.9 --mixup 0.2 --aa rand-m9-mstd0.5-inc1 --reprob 0.5 --remode pixel --output "./experiments/task3/self_attn_v2" -j 8 --amp --native-amp

# V4iT-C (training)

python train.py --dataset "aircraft" --data_dir "/apps/local/shared/CV703/datasets/" --num-classes 100 --model "inception_v4_itc" --pretrained -b 32 --epochs 800 --lr 0.025 --weight-decay 0.0005 --decay-epochs 10 --decay-rate 0.9 --mixup 0.2 --aa rand-m9-mstd0.5-inc1 --reprob 0.5 --remode pixel --output "./experiments/task1/self_attn_v2" -j 8 --amp --native-amp

python train.py --dataset "task2" --data_dir "/apps/local/shared/CV703/datasets/" --num-classes 296 --model "inception_v4_itc" --pretrained -b 32 --epochs 800 --lr 0.025 --weight-decay 0.0005 --decay-epochs 10 --decay-rate 0.9 --mixup 0.2 --aa rand-m9-mstd0.5-inc1 --reprob 0.5 --remode pixel --output "./experiments/task2/self_attn_v2" -j 8 --amp --native-amp

python train.py --dataset "task3" --data_dir "/apps/local/shared/CV703/datasets/FoodX/food_dataset" --num-classes 251 --model "inception_v4_itc" --pretrained -b 32 --epochs 800 --lr 0.025 --weight-decay 0.0005 --decay-epochs 10 --decay-rate 0.9 --mixup 0.2 --aa rand-m9-mstd0.5-inc1 --reprob 0.5 --remode pixel --output "./experiments/task3/self_attn_v2" -j 8 --amp --native-amp
