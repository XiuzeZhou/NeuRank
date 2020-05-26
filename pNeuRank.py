import tensorflow as tf
import numpy as np
import math
from pylab import *
from data import *
from sklearn.model_selection import KFold
from sklearn.metrics import auc,roc_auc_score,precision_recall_curve
import argparse


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run pNeuRank.")
    parser.add_argument('--path', nargs='?', default='datasets/',
                        help='Input data path.')
    parser.add_argument('--data_name', nargs='?', default='Enzyme',
                        help='Choose a dataset.')
    parser.add_argument('--epoches', type=int, default=40,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layers', nargs='?', default='[32]',
                        help="Size of each layer. Note that the first hidden layer is the interaction layer.")
    parser.add_argument('--reg', type=float, default=0.00001,
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--cv', type=int, default=10,
                        help='K-fold Cross Validation.')
    parser.add_argument('--min_loss', type=float, default=0.1,
                        help='The minimum value for the loss function.')
    parser.add_argument('--mode', type=int, default=0,
                        help='the mode for training: 0 -> train for drug-target pairs; 1 -> train for new drugs; 2 -> train for new target')
    return parser.parse_args()


class pNeuRank():
    def __init__(self,               
            drugs_num = None,         # drug number
            targets_num = None,        # target number
            batch_size = 64,          # batch size
            embedding_size = 64,       # embedding size
            hidden_size = [32],       # hiedden layers
            learning_rate = 1e-3,     # learning rate
            lamda_regularizer = 1e-5   # regularization coefficient for L2
            ):
        self.drugs_num = drugs_num
        self.targets_num = targets_num
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.lamda_regularizer = lamda_regularizer

        # loss records
        self.train_loss_records = []   
        self.build_graph()   

        
    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # _________ input data _________
            self.drugs_inputs = tf.compat.v1.placeholder(tf.int32, shape = [None], name='drugs_inputs')
            self.targets_inputs = tf.compat.v1.placeholder(tf.int32, shape = [None,2], name='targets_inputs')
            
            # _________ variables _________
            self.weights = self._initialize_weights()
            
            # _________ train _____________
            self.y_ui = self.inference(drugs_inputs=self.drugs_inputs, targets_inputs=self.targets_inputs[:,0])
            self.y_uj = self.inference(drugs_inputs=self.drugs_inputs, targets_inputs=self.targets_inputs[:,1])
            self.loss_train = self.loss_function(y_ui=self.y_ui, y_uj=self.y_uj, lamda_regularizer=self.lamda_regularizer)
            self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss_train) 

            # _________ prediction _____________
            self.predictions = self.inference(drugs_inputs=self.drugs_inputs,targets_inputs=self.targets_inputs[:,1])
        
            # variables init
            self.saver = tf.compat.v1.train.Saver() 
            init = tf.compat.v1.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)
    
    
    def _init_session(self):
        # adaptively growing memory
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.compat.v1.Session(config=config)
    
    
    def _initialize_weights(self):
        all_weights = dict()

        # -----embedding layer------
        all_weights['embedding_drugs'] = tf.Variable(tf.random.normal([self.drugs_num, self.embedding_size], 0, 0.1),name='embedding_drugs')
        all_weights['embedding_targets'] = tf.Variable(tf.random.normal([self.targets_num, self.embedding_size], 0, 0.1),name='embedding_targets') 
        
        # ------hidden layer------
        all_weights['weight_0'] = tf.Variable(tf.random.normal([self.embedding_size,self.hidden_size[0]], 0.0, 0.1),name='weight_0')
        all_weights['bias_0'] = tf.Variable(tf.zeros([self.hidden_size[0]]), name='bias_0')
        
        # ------output layer-----
        all_weights['weight_n'] = tf.Variable(tf.random.normal([self.hidden_size[-1], 1], 0, 0.1), name='weight_n')
        all_weights['bias_n'] = tf.Variable(tf.zeros([1]), name='bias_n')

        return all_weights
        
    
    def train(self, data_sequence):
        train_size = len(data_sequence)
        
        np.random.shuffle(data_sequence)
        batch_size = self.batch_size
        total_batch = math.ceil(train_size/batch_size)

        for batch in range(total_batch):
            start = (batch*batch_size)% train_size
            end = min(start+batch_size, train_size)
            data_array = np.array(data_sequence[start:end])

            loss_val=self.fit(X=data_array)
            self.train_loss_records.append(loss_val)
            
        return self.train_loss_records

        
    def inference(self, drugs_inputs, targets_inputs):
        embed_drugs = tf.reshape(tf.nn.embedding_lookup(self.weights['embedding_drugs'], drugs_inputs),
                                 shape=[-1, self.embedding_size])
        embed_targets = tf.reshape(tf.nn.embedding_lookup(self.weights['embedding_targets'], targets_inputs),
                                 shape=[-1, self.embedding_size])
            
        layer0 = tf.nn.relu(tf.matmul(embed_targets*embed_drugs, self.weights['weight_0']) + self.weights['bias_0'])
        y_ = tf.matmul(layer0,self.weights['weight_n']) + self.weights['bias_n']
        return y_         
        
        
    def fit(self, X):
        # X: input data
        feed_dict = {self.drugs_inputs: X[:,0], self.targets_inputs: X[:,1:3]}  
        loss, opt = self.sess.run([self.loss_train,self.train_op], feed_dict=feed_dict)
        return loss
        
        
    def loss_function(self, y_ui, y_uj, lamda_regularizer=1e-5):   
        y_uij = tf.sigmoid(y_ui-y_uj)
        bpr_loss = -tf.reduce_sum(tf.math.log(y_uij),axis=0)
        cost = bpr_loss
        if lamda_regularizer>0:
            regularizer_1 = tf.contrib.layers.l2_regularizer(lamda_regularizer)
            regularization = regularizer_1(
                self.weights['embedding_drugs']) + regularizer_1(
                self.weights['embedding_targets'])+ regularizer_1(
                self.weights['weight_0']) + regularizer_1(
                self.weights['weight_n'])
            cost = bpr_loss + regularization

        return  cost  
        
        
    def evaluate(self, X, labels):
        drugs_inputs = X[:,0]
        targets_inputs = X
        feed_dict = {self.drugs_inputs: drugs_inputs, self.targets_inputs: targets_inputs}  
        score = self.sess.run([self.predictions], feed_dict=feed_dict)       
        y_pred = np.reshape(score,(-1))
        
        auc_score = roc_auc_score(y_true=labels,y_score=y_pred)
        precision, recall, thresholds = precision_recall_curve(labels, y_pred)
        aupr_score = auc(recall, precision)
        return auc_score, aupr_score

    
# train for model
def train(model, data_list, drugs_num, targets_num, epoches=40, cv=10, sample_size=4, min_loss=0.1, mode=0): 
    # k-fold cross validation
    kf = KFold(n_splits=cv, shuffle=True)
    data_mat = sequence2mat(sequence=data_list, N=drugs_num, M=targets_num)
    
    cv_auc_list, cv_aupr_list = [],[]
    if mode==0: # train for drug-target pairs
        print('Train for drug-target pairs:')
        instances_list = []
        [instances_list.append([d,t,data_mat[d,t]]) for d in range(drugs_num) for t in range(targets_num)]
        for train_ids, test_ids in kf.split(instances_list):
            train_list = np.array(instances_list)[train_ids]
            test_list = np.array(instances_list)[test_ids][:,:2]
            test_labels = np.array(instances_list)[test_ids][:,-1]
            train_mat = sequence2mat(sequence=train_list, N=drugs_num, M=targets_num)# train data : user-item matrix

            auc_score, aupr_score = model.evaluate(X=np.array(test_list), labels=test_labels)
            print('Init: AUC = %.4f, AUPR=%.4f' %(auc_score, aupr_score))

            auc_list, aupr_list = [],[]
            auc_list.append(auc_score)
            aupr_list.append(aupr_score)
            for epoch in range(epoches):
                data_sequence = generate_data(train_mat=train_mat, sample_size=sample_size, mode=1)
                loss_records = model.train(data_sequence=data_sequence)
                auc_score, aupr_score = model.evaluate(X=np.array(test_list), labels=test_labels)
                auc_list.append(auc_score)
                aupr_list.append(aupr_score)
                print('epoch=%d, loss=%.4f, AUC=%.4f, AUPR=%.4f' %(epoch,loss_records[-1],auc_score, aupr_score))

                if loss_records[-1]<min_loss:
                    break
            cv_auc_list.append(auc_list[-1])
            cv_aupr_list.append(aupr_list[-1])

    elif mode==1: # train for new drugs
        print('Train for new drugs:')
        for train_ids, test_ids in kf.split(range(drugs_num)):
            instances_train = []
            [instances_train.append([d,t,data_mat[d,t]]) for d in train_ids for t in range(targets_num)]
            instances_test = []
            [instances_test.append([d,t,data_mat[d,t]]) for d in test_ids for t in range(targets_num)]

            train_list = np.array(instances_train)
            test_list = np.array(instances_test)[:,:2]
            test_labels = np.array(instances_test)[:,-1]
            train_mat = sequence2mat(sequence=train_list, N=drugs_num, M=targets_num)# train data : user-item matrix

            auc_score, aupr_score = model.evaluate(X=np.array(test_list), labels=test_labels)
            print('Init: AUC = %.4f, AUPR=%.4f' %(auc_score, aupr_score))

            auc_list, aupr_list = [],[]
            auc_list.append(auc_score)
            aupr_list.append(aupr_score)
            for epoch in range(epoches):
                data_sequence = generate_data(train_mat=train_mat, sample_size=sample_size, mode=1)
                loss_records = model.train(data_sequence=data_sequence)
                auc_score, aupr_score = model.evaluate(X=np.array(test_list), labels=test_labels)
                auc_list.append(auc_score)
                aupr_list.append(aupr_score)
                print('epoch=%d, loss=%.4f, AUC=%.4f, AUPR=%.4f' %(epoch,loss_records[-1],auc_score, aupr_score))

                if loss_records[-1]<min_loss:
                    break
            cv_auc_list.append(auc_list[-1])
            cv_aupr_list.append(aupr_list[-1])
            
    elif mode==2: # train for new targets
        print('Train for new targets:')
        for train_ids, test_ids in kf.split(range(targets_num)):
            instances_train = []
            [instances_train.append([d,t,data_mat[d,t]]) for d in range(drugs_num) for t in train_ids]
            instances_test = []
            [instances_test.append([d,t,data_mat[d,t]]) for d in range(drugs_num) for t in test_ids]

            train_list = np.array(instances_train)
            test_list = np.array(instances_test)[:,:2]
            test_labels = np.array(instances_test)[:,-1]
            train_mat = sequence2mat(sequence=train_list, N=drugs_num, M=targets_num)# train data : user-item matrix

            auc_score, aupr_score = model.evaluate(X=np.array(test_list), labels=test_labels)
            print('Init: AUC = %.4f, AUPR=%.4f' %(auc_score, aupr_score))

            auc_list, aupr_list = [],[]
            auc_list.append(auc_score)
            aupr_list.append(aupr_score)
            for epoch in range(epoches):
                data_sequence = generate_data(train_mat=train_mat, sample_size=sample_size, mode=1)
                loss_records = model.train(data_sequence=data_sequence)
                auc_score, aupr_score = model.evaluate(X=np.array(test_list), labels=test_labels)
                auc_list.append(auc_score)
                aupr_list.append(aupr_score)
                print('epoch=%d, loss=%.4f, AUC=%.4f, AUPR=%.4f' %(epoch,loss_records[-1],auc_score, aupr_score))

                if loss_records[-1]<min_loss:
                    break
            cv_auc_list.append(auc_list[-1])
            cv_aupr_list.append(aupr_list[-1])                             
    
    print('Mean AUC=%.4f, AUPR=%.4f' %(np.mean(cv_auc_list),np.mean(cv_aupr_list)))


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    data_name = args.data_name
    embedding_size = args.num_factors
    hidden_size = eval(args.layers)
    lamda_regularizer = args.reg
    sample_size = args.num_neg
    learning_rate = args.lr
    batch_size = args.batch_size
    cv = args.cv
    epoches = args.epoches
    min_loss = args.min_loss
    mode = args.mode

    data_dir = path + data_name + '.txt'
    drugs_num, targets_num, data_list, _, _ = load_data(file_dir=data_dir)
    print(data_name + ': N=%d, M=%d' %(drugs_num, targets_num))
    
    # build model
    model = pNeuRank(drugs_num = drugs_num,
               targets_num = targets_num,
               batch_size = batch_size,
               embedding_size = embedding_size,
               hidden_size = hidden_size,
               learning_rate = learning_rate,
               lamda_regularizer=lamda_regularizer
               )
    
    train(model = model, 
        data_list = data_list, 
        drugs_num = drugs_num, 
        targets_num = targets_num, 
        epoches = epoches, 
        cv = cv,
        sample_size = sample_size,
        min_loss = min_loss,
        mode = mode)