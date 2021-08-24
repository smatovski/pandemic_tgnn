import torch
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pandas as pd

import os
    
def read_meta_datasets(window=7):
    os.chdir("../data")
    meta_graphs = []
    meta_features = []
    meta_y = []

    suppliersnn = pd.read_excel("SC_Graph_synthetic.xlsx", sheet_name="Suppliers")
    manufacturersnn = pd.read_excel("SC_Graph_synthetic.xlsx", sheet_name="Manufacturers")
    distributorsnn = pd.read_excel("SC_Graph_synthetic.xlsx", sheet_name="Distributors")
    customersnn = pd.read_excel("SC_Graph_synthetic.xlsx", sheet_name="Customers")
    suppliers = normalize(suppliersnn)
    manufacturers = normalize(manufacturersnn)
    distributors = normalize(distributorsnn)
    customers = normalize(customersnn)
    
    Gs = generate_graphs_tmp()
     
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    meta_graphs.append(gs_adj)

    features = generate_new_features(Gs ,window, suppliers, manufacturers, distributors, customers)

    meta_features.append(features)

    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        y[i].append(suppliers['Supplier_001_Capacity'][i])
        y[i].append(suppliers['Supplier_002_Capacity'][i])
        y[i].append(suppliers['Supplier_003_Capacity'][i])
        y[i].append(suppliers['Supplier_004_Capacity'][i])
        y[i].append(suppliers['Supplier_005_Capacity'][i])
        y[i].append(manufacturers['Manufacturer_001_Capacity'][i])
        y[i].append(manufacturers['Manufacturer_002_Capacity'][i])
        y[i].append(manufacturers['Manufacturer_003_Capacity'][i])
        y[i].append(manufacturers['Manufacturer_004_Capacity'][i])
        y[i].append(manufacturers['Manufacturer_005_Capacity'][i])
        y[i].append(distributors['Distributor_001_Inventory'][i])
        y[i].append(distributors['Distributor_002_Inventory'][i])
        y[i].append(distributors['Distributor_003_Inventory'][i])
        y[i].append(distributors['Distributor_004_Inventory'][i])
        y[i].append(distributors['Distributor_005_Inventory'][i])
        y[i].append(distributors['Distributor_006_Inventory'][i])
        y[i].append(customers['Customer_001_Inventory'][i])
        y[i].append(customers['Customer_002_Inventory'][i])
        y[i].append(customers['Customer_003_Inventory'][i])
        y[i].append(customers['Customer_004_Inventory'][i])

    meta_y.append(y)

    os.chdir("../code")

    return meta_graphs, meta_features, meta_y

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if (max_value - min_value) != 0:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        else:
            result[feature_name] = df[feature_name]
            
    return result   
    
def generate_graphs_tmp():
    Gs = []
    d = pd.read_excel("SC_Graph_synthetic.xlsx", sheet_name="Edge")
    del d['Time']
    nd = normalize(d)

    for Graphs in nd.iterrows():
        G = nx.DiGraph()
        for i in range(0,len(Graphs[1]),3):
            G.add_edge(Graphs[1][i], Graphs[1][i+1], weight=Graphs[1][i+2])
        Gs.append(G)
        
    return Gs

def generate_new_features(Gs, window, suppliers, manufacturers, distributors, customers):
    """
    Generate node features
    Features[1] contains the features corresponding to y[1]
    e.g. if window = 7, features[7]= day0:day6, y[7] = day7
    if the window reaches before 0, everything is 0, so features[3] = [0,0,0,0,day0,day1,day2], y[3] = day3
    """
    features = list()
   
    #--- one hot encoded the region
    #departments_name_to_id = dict()
    #for node in nodes:
    #departments_name_to_id[node] = len(departments_name_to_id)
    #n_departments = len(departments_name_to_id)
        
    #print(n_departments)
    for idx,G in enumerate(Gs):
    #  Features = population, coordinates, d past cases, one hot region
            
        H = np.zeros([G.number_of_nodes(),window]) #+3+n_departments])#])#])
        
        ### enumarate because H[i] and labs[node] are not aligned
        #---- Past cases      
        if(idx < window):# idx-1 goes before the start of the labels
            H[0,(window-idx):(window)] = suppliers['Supplier_001_Capacity'][0:(idx)]
            H[1,(window-idx):(window)] = suppliers['Supplier_002_Capacity'][0:(idx)]
            H[2,(window-idx):(window)] = suppliers['Supplier_003_Capacity'][0:(idx)]
            H[3,(window-idx):(window)] = suppliers['Supplier_004_Capacity'][0:(idx)]
            H[4,(window-idx):(window)] = suppliers['Supplier_005_Capacity'][0:(idx)]
            H[5,(window-idx):(window)] = manufacturers['Manufacturer_001_Capacity'][0:(idx)]
            H[6,(window-idx):(window)] = manufacturers['Manufacturer_002_Capacity'][0:(idx)]
            H[7,(window-idx):(window)] = manufacturers['Manufacturer_003_Capacity'][0:(idx)]
            H[8,(window-idx):(window)] = manufacturers['Manufacturer_004_Capacity'][0:(idx)]
            H[9,(window-idx):(window)] = manufacturers['Manufacturer_005_Capacity'][0:(idx)]
            H[10,(window-idx):(window)] = distributors['Distributor_001_Inventory'][0:(idx)]
            H[11,(window-idx):(window)] = distributors['Distributor_002_Inventory'][0:(idx)]
            H[12,(window-idx):(window)] = distributors['Distributor_003_Inventory'][0:(idx)]
            H[13,(window-idx):(window)] = distributors['Distributor_004_Inventory'][0:(idx)]
            H[14,(window-idx):(window)] = distributors['Distributor_005_Inventory'][0:(idx)]
            H[15,(window-idx):(window)] = distributors['Distributor_006_Inventory'][0:(idx)]
            H[16,(window-idx):(window)] = customers['Customer_001_Inventory'][0:(idx)]
            H[17,(window-idx):(window)] = customers['Customer_002_Inventory'][0:(idx)]
            H[18,(window-idx):(window)] = customers['Customer_003_Inventory'][0:(idx)]
            H[19,(window-idx):(window)] = customers['Customer_004_Inventory'][0:(idx)]
                    
        elif idx >= window:
            H[0,0:(window)] = suppliers['Supplier_001_Capacity'][(idx-window):(idx)]
            H[1,0:(window)] = suppliers['Supplier_002_Capacity'][(idx-window):(idx)]
            H[2,0:(window)] = suppliers['Supplier_003_Capacity'][(idx-window):(idx)]
            H[3,0:(window)] = suppliers['Supplier_004_Capacity'][(idx-window):(idx)]
            H[4,0:(window)] = suppliers['Supplier_005_Capacity'][(idx-window):(idx)]
            H[5,0:(window)] = manufacturers['Manufacturer_001_Capacity'][(idx-window):(idx)]
            H[6,0:(window)] = manufacturers['Manufacturer_002_Capacity'][(idx-window):(idx)]
            H[7,0:(window)] = manufacturers['Manufacturer_003_Capacity'][(idx-window):(idx)]
            H[8,0:(window)] = manufacturers['Manufacturer_004_Capacity'][(idx-window):(idx)]
            H[9,0:(window)] = manufacturers['Manufacturer_005_Capacity'][(idx-window):(idx)]
            H[10,0:(window)] = distributors['Distributor_001_Inventory'][(idx-window):(idx)]
            H[11,0:(window)] = distributors['Distributor_002_Inventory'][(idx-window):(idx)]
            H[12,0:(window)] = distributors['Distributor_003_Inventory'][(idx-window):(idx)]
            H[13,0:(window)] = distributors['Distributor_004_Inventory'][(idx-window):(idx)]
            H[14,0:(window)] = distributors['Distributor_005_Inventory'][(idx-window):(idx)]
            H[15,0:(window)] = distributors['Distributor_006_Inventory'][(idx-window):(idx)]
            H[16,0:(window)] = customers['Customer_001_Inventory'][(idx-window):(idx)]
            H[17,0:(window)] = customers['Customer_002_Inventory'][(idx-window):(idx)]
            H[18,0:(window)] = customers['Customer_003_Inventory'][(idx-window):(idx)]
            H[19,0:(window)] = customers['Customer_004_Inventory'][(idx-window):(idx)]
            
        features.append(H)
        
    return features


def generate_new_batches(Gs, features, y, idx, graph_window, shift, batch_size, device, test_sample):
    """
    Generate batches for graphs for MPNN
    """

    N = len(idx)
    n_nodes = Gs[0].shape[0]
    #n_nodes = Gs[0].number_of_nodes()
  
    adj_lst = list()
    features_lst = list()
    y_lst = list()

    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i+batch_size, N)-i)*graph_window*n_nodes
        step = n_nodes*graph_window

        adj_tmp = list()
        features_tmp = np.zeros((n_nodes_batch, features[0].shape[1]))

        y_tmp = np.zeros((min(i+batch_size, N)-i)*n_nodes)

        #fill the input for each batch
        for e1,j in enumerate(range(i, min(i+batch_size, N) )):
            val = idx[j]

            # Feature[10] containes the previous 7 cases of y[10]
            for e2,k in enumerate(range(val-graph_window+1,val+1)):
                
                adj_tmp.append(Gs[k-1].T)  
                # each feature has a size of n_nodes
                features_tmp[(e1*step+e2*n_nodes):(e1*step+(e2+1)*n_nodes),:] = features[k]#-features[val-graph_window-1]
            
            
            if(test_sample>0):
                #--- val is by construction less than test sample
                if(val+shift<test_sample):
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
                    
                else:
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val]
                        
                        
            else:
                y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
        
        adj_tmp = sp.block_diag(adj_tmp)
        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_tmp).to(device))
        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append(torch.FloatTensor(y_tmp).to(device))

    return adj_lst, features_lst, y_lst


def generate_batches_lstm(n_nodes, y, idx, window, shift, batch_size, device,test_sample):
    """
    Generate batches for graphs for the LSTM
    """
    N = len(idx)
    features_lst = list()
    y_lst = list()
    adj_fake = list()
    
    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i+batch_size, N)-i)*n_nodes*1
        #step = n_nodes#*window
        step = n_nodes*1

        adj_tmp = list()
        features_tmp = np.zeros((window, n_nodes_batch))#features.shape[1]))
        
        y_tmp = np.zeros((min(i+batch_size, N)-i)*n_nodes)
        
        for e1,j in enumerate(range(i, min(i+batch_size, N))):
            val = idx[j]
            
            # keep the past information from val-window until val-1
            for e2,k in enumerate(range(val-window,val)):
               
                if(k==0): 
                    features_tmp[e2, (e1*step):(e1*step+n_nodes)] = np.zeros([n_nodes])#features#[k]
                else:
                    features_tmp[e2, (e1*step):(e1*step+n_nodes)] = np.array(y[k])#.reshape([n_nodes,1])#

            if(test_sample>0):
                # val is by construction less than test sample
                if(val+shift<test_sample):
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
                else:
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val]
                        
            else:
         
                y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]       
         
        adj_fake.append(0)
        
        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append( torch.FloatTensor(y_tmp).to(device))
        
    return adj_fake, features_lst, y_lst




def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

