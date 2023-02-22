# To reproduce the results, please run the following commands

python eval.py --dataset "aircraft" --data_dir "/apps/local/shared/CV703/datasets/" --num-classes 100 --model "inception_v4_itc" --pretrained -b 32 --epochs 800 --lr 0.025 --weight-decay 0.0005 --decay-epochs 10 --decay-rate 0.9 --mixup 0.2 --aa rand-m9-mstd0.5-inc1 --reprob 0.5 --remode pixel --output "./experiments/task1/self_attn_v2" -j 8 --amp --native-amp --resume "/l/users/fadillah.maani/CV703B/A01/V4iT (Concat)/Aircraft/model_best.pth.tar"
# Test: [ 104/104]  Time: 0.038 (0.160)  Loss:  1.2734 (0.3487)  Acc@1: 80.0000 (94.2994)  Acc@5: 80.0000 (98.3798)

python eval.py --dataset "task2" --data_dir "/apps/local/shared/CV703/datasets/" --num-classes 296 --model "inception_v4_itc" --pretrained -b 32 --epochs 800 --lr 0.025 --weight-decay 0.0005 --decay-epochs 10 --decay-rate 0.9 --mixup 0.2 --aa rand-m9-mstd0.5-inc1 --reprob 0.5 --remode pixel --output "./experiments/task2/self_attn_v2" -j 8 --amp --native-amp --resume "/l/users/fadillah.maani/CV703B/A01/V4iT (Concat)/Combine/model_best.pth.tar"
# Test: [ 355/355]  Time: 0.063 (0.137)  Loss:  0.2727 (0.4314)  Acc@1: 100.0000 (94.9710)  Acc@5: 100.0000 (99.2351)

python eval.py --dataset "task3" --data_dir "/apps/local/shared/CV703/datasets/FoodX/food_dataset" --num-classes 251 --model "inception_v4_itc" --pretrained -b 32 --epochs 800 --lr 0.025 --weight-decay 0.0005 --decay-epochs 10 --decay-rate 0.9 --mixup 0.2 --aa rand-m9-mstd0.5-inc1 --reprob 0.5 --remode pixel --output "./experiments/task3/self_attn_v2" -j 8 --amp --native-amp --resume "/l/users/fadillah.maani/CV703B/A01/V4iT (Concat)/Food/checkpoint-483.pth.tar"
# Test: [ 374/374]  Time: 0.098 (0.132)  Loss:  1.6562 (1.2688)  Acc@1: 69.2308 (71.8026)  Acc@5: 84.6154 (90.4202)