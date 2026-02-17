gcnsa:
python train.py --dataset ClinTox --dropout 0.4 --hd 16 --K 1 --seed 42 --wd 5e-3 --lr 0.001 --epsilon 0.9 --r 3 --trainsplit 0.80 --epochs 500

hgnn:
python train.py --dataset clintox --epochs 100 --lr 0.0001 --batch_size 128 --depth 5
