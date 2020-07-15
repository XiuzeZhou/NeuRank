import numpy as np

# Load data from file_dir
def load_data(file_dir):
    # output: 
    # N: the number of user;
    # M: the number of item
    # data: the list of rating information
    drug_ids_dict, target_ids_dict = {},{}
    N,M,d_idx,t_idx = 0,0,0,0 # N: the number of drug; M: the number of target
    data = []
    f = open(file_dir)
    for line in f.readlines():
        d, t = line.split()
        d = d.replace(':','')
        if d not in drug_ids_dict:
            drug_ids_dict[d]=d_idx
            d_idx+=1
        if t not in target_ids_dict:
            target_ids_dict[t]=t_idx
            t_idx+=1
        data.append([drug_ids_dict[d],target_ids_dict[t],1])
    
    f.close()
    N = d_idx
    M = t_idx

    return N, M, data, drug_ids_dict, target_ids_dict


# Convert the list data to array
def sequence2mat(sequence, N, M):
    # input:
    # sequence: the list of rating information
    # N: row number, i.e. the number of users
    # M: column number, i.e. the number of items
    # output:
    # mat: user-item rating matrix
    records_array = np.array(sequence)
    mat = np.zeros([N,M])
    row = records_array[:,0].astype(int)
    col = records_array[:,1].astype(int)
    values = records_array[:,2].astype(np.float32)
    mat[row,col]=values
    
    return mat


# Sample for imbalanced data
def generate_data(train_mat, sample_size=4, mode=0):
    drugs_num,targets_num = train_mat.shape
    data = []
    
    if mode==0:
        for d in range(drugs_num):
            positive_targets = np.where(train_mat[d,:]>0)[0] # the observed interactions with drug d

            for target0 in positive_targets:
                data.append([d,target0,1])
                for _ in range(sample_size):
                    target1 = np.random.randint(targets_num)
                    while (target1 in positive_targets) or (train_mat[d,target1]!=0):
                        target1 = np.random.randint(targets_num)
                    data.append([d,target1,0])

    else:
        for d in range(drugs_num):
            positive_targets = np.where(train_mat[d,:]>0)[0] # the observed interactions with drug d

            for target0 in positive_targets:
                for _ in range(sample_size):
                    target1 = np.random.randint(targets_num)
                    while (target1 in positive_targets) or (train_mat[d,target1]!=0):
                        target1 = np.random.randint(targets_num)
                    data.append([d,target0,target1])
                    
    return data
