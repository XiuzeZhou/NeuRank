# NeuRank
NeuRank: Learning to Ranking with Neural Networks for Drug-Target Interaction Prediction

- Run NeuRank:

$ python NeuRank.py --path datasets/ --data_name Enzyme --epoches 40 --batch_size 64 --num_factors 64 --layers [32,16] --reg 0.00001 --num_neg 4 --lr 0.001 --min_loss 0.01 --cv 10 --mode 0

- Run NeuRanks:

$ python NeuRanks.py --path datasets/ --data_name Enzyme --epoches 100 --batch_size 64 --num_factors 64 --layers [32,16] --reg [0.00001,0.000001,0.000001] --num_neg 4 --lr 0.001 --min_loss 0.01 --cv 10 --mode 0

- Run pNeuRank:

$ python pNeuRank.py --path datasets/ --data_name Enzyme --epoches 40 --batch_size 64 --num_factors 64 --layers [32] --reg 0.00001 --num_neg 4 --lr 0.001 --min_loss 0.1 --cv 10 --mode 0
