# NeuRank
NeuRank: Learning to Ranking with Neural Networks for Drug-Target Interaction Prediction

- **Run NeuRank**:

$ python NeuRank.py --path datasets/ --data_name Enzyme --epoches 40 --batch_size 64 --num_factors 64 --layers [32,16] --reg 0.00001 --num_neg 4 --lr 0.001 --min_loss 0.01 --cv 10 --mode 0

- **Run NeuRanks**:

$ python NeuRanks.py --path datasets/ --data_name Enzyme --epoches 100 --batch_size 64 --num_factors 64 --layers [32,16] --reg [0.00001,0.000001,0.000001] --num_neg 4 --lr 0.001 --min_loss 0.01 --cv 10 --mode 0

- **Run pNeuRank**:

$ python pNeuRank.py --path datasets/ --data_name Enzyme --epoches 40 --batch_size 64 --num_factors 64 --layers [32] --reg 0.00001 --num_neg 4 --lr 0.001 --min_loss 0.1 --cv 10 --mode 0

## Parameter description：
- path：Input data path.
- data_name：Name of dataset
- epoches：Number of epoches.
- batch_size：Batch size.
- num_factors：Embedding size.
- layers：Size of each layer. Note that the first hidden layer is the interaction layer.
- reg: Regularization for user and item embeddings.
- num_neg: Number of negative instances to pair with a positive instance.
- lr: Learning rate.
- min_loss: The minimum value for stopping loss function.
- cv: K-fold Cross Validation.
- mode: the mode for training: 0 -> train for drug-target pairs; 1 -> train for new drugs; 2 -> train for new target
