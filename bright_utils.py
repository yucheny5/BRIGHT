import random
import config
import pickle
from pyrwr.rwr import RWR
import numpy as np
import scipy
import scipy.spatial as sp
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

'''
Dataset  preprocessing
'''
'''
This function can be used for Cora and DBLP two attribute network datasets.
'''
def ori2norm_Cora():
    data_name = 'Data/DBLP/ori_data/new-ACM-DBLP'
    data = np.load('%s.npz' % data_name)
    edge_index1, edge_index2 = data['edge_index1'], data['edge_index2']
    gnd = data['gnd']
    norm_g1_edge = []
    norm_g2_edge = []
    for i in range(len(edge_index1[0])):
        if (edge_index1[0][i], edge_index1[1][i]) not in norm_g1_edge:
            norm_g1_edge.append((edge_index1[0][i], edge_index1[1][i]))
        if (edge_index1[1][i], edge_index1[0][i]) not in norm_g1_edge:
            norm_g1_edge.append((edge_index1[1][i], edge_index1[0][i]))
    for i in range(len(edge_index2[0])):
        if (edge_index2[0][i], edge_index2[1][i]) not in norm_g2_edge:
            norm_g2_edge.append((edge_index2[0][i], edge_index2[1][i]))
        if (edge_index2[1][i], edge_index2[0][i]) not in norm_g2_edge:
            norm_g2_edge.append((edge_index2[1][i], edge_index2[0][i]))
    norm_g1_file = open("Data/DBLP/norm_data/network1.tsv", 'w')
    norm_g2_file = open("Data/DBLP/norm_data/network2.tsv", 'w')
    for edge in norm_g1_edge:
        norm_g1_file.write(str(list(edge)[0]) + " " + str(list(edge)[1]) + " " + "1" + '\n')
    for edge in norm_g2_edge:
        norm_g2_file.write(str(list(edge)[0]) + " " + str(list(edge)[1]) + " " + "1" + '\n')
    grd_truth_file = open("Data/DBLP/norm_data/grd.tsv", 'w')
    for node1, node2 in gnd:
        grd_truth_file.write(str(node1) + " " + str(node2) + '\n')
    norm_g1_file.close()
    norm_g2_file.close()
    grd_truth_file.close()



def ori2norm_D2A():
    f1 = open("Data/D2A/ori_data/dblp.number", 'r')
    f2 = open("Data/D2A/ori_data/acm.number", 'r')
    seed_file = open("Data/D2A/ori_data/D2A.align", 'r')
    ori_g1 = [set(), []]
    ori_g2 = [set(), []]
    for line in f1.readlines():
        edge = line.strip('\n').split()
        ori_g1[0].add(int(edge[0]))
        ori_g1[0].add(int(edge[1]))
        if (int(edge[0]), int(edge[1])) not in ori_g1[1]:
            ori_g1[1].append((int(edge[0]), int(edge[1])))
        if (int(edge[1]), int(edge[0])) not in ori_g1[1]:
            ori_g1[1].append((int(edge[1]), int(edge[0])))
    for line in f2.readlines():
        edge = line.strip('\n').split()
        ori_g2[0].add(int(edge[0]))
        ori_g2[0].add(int(edge[1]))
        if (int(edge[0]), int(edge[1])) not in ori_g2[1]:
            ori_g2[1].append((int(edge[0]), int(edge[1])))
        if (int(edge[1]), int(edge[0])) not in ori_g2[1]:
            ori_g2[1].append((int(edge[1]), int(edge[0])))
    ori_g1[0] = list(ori_g1[0])
    ori_g2[0] = list(ori_g2[0])
    random.shuffle(ori_g1[0])
    random.shuffle(ori_g2[0])
    map_g1 = {}
    map_g2 = {}
    for i in range(len(ori_g1[0])):
        map_g1[ori_g1[0][i]] = i
    for i in range(len(ori_g2[0])):
        map_g2[ori_g2[0][i]] = i
    norm_g1_edge = []
    norm_g2_edge = []
    for edge in ori_g1[1]:
        norm_g1_edge.append([map_g1[list(edge)[0]], map_g1[list(edge)[1]]])
    for edge in ori_g2[1]:
        norm_g2_edge.append([map_g2[list(edge)[0]], map_g2[list(edge)[1]]])
    f1.close()
    f2.close()
    grd_truth = []
    for line in seed_file.readlines():
        pair = line.strip('\n').split()
        grd_truth.append([map_g1[int(pair[0])], map_g2[int(pair[1])]])
    seed_file.close()

    norm_g1_file = open("Data/D2A/norm_data/network1.tsv", 'w')
    norm_g2_file = open("Data/D2A/norm_data/network2.tsv", 'w')
    for edge in norm_g1_edge:
        norm_g1_file.write(str(edge[0]) + " " + str(edge[1]) + " " + "1" + '\n')
    for edge in norm_g2_edge:
        norm_g2_file.write(str(edge[0]) + " " + str(edge[1]) + " " + "1" + '\n')
    grd_truth_file = open("Data/D2A/norm_data/grd.tsv", 'w')
    for pair in grd_truth:
        grd_truth_file.write(str(pair[0]) + " " + str(pair[1]) + '\n')
    norm_g1_file.close()
    norm_g2_file.close()
    grd_truth_file.close()


def ori2norm_F2T():

    f1 = open("Data/F2T/ori_data/foursqure.number", 'r')
    f2 = open("Data/F2T/ori_data/twitter.number", 'r')
    seed_file = open("Data/F2T/ori_data/groundtruth.number", 'r')
    ori_g1 = [set(), []]
    ori_g2 = [set(), []]

    for line in f1.readlines():
        edge = line.strip('\n').split()
        ori_g1[0].add(int(edge[0]))
        ori_g1[0].add(int(edge[1]))
        if (int(edge[0]), int(edge[1])) not in ori_g1[1]:
            ori_g1[1].append((int(edge[0]), int(edge[1])))
        if (int(edge[1]), int(edge[0])) not in ori_g1[1]:
            ori_g1[1].append((int(edge[1]), int(edge[0])))
    for line in f2.readlines():
        edge = line.strip('\n').split()
        ori_g2[0].add(int(edge[0]))
        ori_g2[0].add(int(edge[1]))
        if (int(edge[0]), int(edge[1])) not in ori_g2[1]:
            ori_g2[1].append((int(edge[0]), int(edge[1])))
        if (int(edge[1]), int(edge[0])) not in ori_g2[1]:
            ori_g2[1].append((int(edge[1]), int(edge[0])))
    ori_g1[0] = list(ori_g1[0])
    ori_g2[0] = list(ori_g2[0])
    random.shuffle(ori_g1[0])
    random.shuffle(ori_g2[0])
    map_g1 = {}
    map_g2 = {}
    for i in range(len(ori_g1[0])):
        map_g1[ori_g1[0][i]] = i
    for i in range(len(ori_g2[0])):
        map_g2[ori_g2[0][i]] = i
    norm_g1_edge = []
    norm_g2_edge = []
    for edge in ori_g1[1]:
        norm_g1_edge.append([map_g1[list(edge)[0]], map_g1[list(edge)[1]]])
    for edge in ori_g2[1]:
        norm_g2_edge.append([map_g2[list(edge)[0]], map_g2[list(edge)[1]]])
    f1.close()
    f2.close()
    grd_truth = []
    for line in seed_file.readlines():
        node = line.strip('\n')
        grd_truth.append([map_g1[int(node)], map_g2[int(node)]])
    seed_file.close()

    norm_g1_file = open("Data/F2T/norm_data/network1.tsv", 'w')
    norm_g2_file = open("Data/F2T/norm_data/network2.tsv", 'w')
    for edge in norm_g1_edge:
        norm_g1_file.write(str(edge[0]) + " " + str(edge[1]) + " " + "1" + '\n')
    for edge in norm_g2_edge:
        norm_g2_file.write(str(edge[0]) + " " + str(edge[1]) + " " + "1" + '\n')
    grd_truth_file = open("Data/F2T/norm_data/grd.tsv", 'w')
    for pair in grd_truth:
        grd_truth_file.write(str(pair[0]) + " " + str(pair[1]) + '\n')
    norm_g1_file.close()
    norm_g2_file.close()
    grd_truth_file.close()

'''
split data
'''
def split_data(ratio):
    f = open(config.grd_truth_file)
    grd_truth = []
    for line in f.readlines():
        pair = line.strip('\n').split()
        grd_truth.append([int(pair[0]), int(pair[1])])
    f.close()
    total = len(grd_truth)
    random.shuffle(grd_truth)
    train = grd_truth[0:int(total*ratio)]
    test = grd_truth[int(total*ratio):total]
    train_file = open(config.train_file + str(ratio) +".pkl", 'wb')
    test_file = open(config.test_file + str(ratio) + ".pkl", 'wb')
    pickle.dump(train, train_file)
    pickle.dump(test, test_file)
    train_file.close()
    test_file.close()
    seed1 = []
    seed2 = []
    seed_file_1 = open(config.seed_file1 + str(ratio) + ".pkl", 'wb')
    seed_file_2 = open(config.seed_file2 + str(ratio) + ".pkl", 'wb')
    for pair in train:
        seed1.append(pair[0])
        seed2.append(pair[1])
    pickle.dump(seed1, seed_file_1)
    pickle.dump(seed2, seed_file_2)
    seed_file_1.close()
    seed_file_2.close()

"""
generate candidate seeds for unsupervised setting with attribute similarity
"""
def get_candi_seed(ratio=0):
    data = np.load('%s.npz' % config.numpy_file)
    g1_feat, g2_feat = data['x1'], data['x2']
    sim = cosine_similarity(g1_feat, g2_feat)
    pair_set = []
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            pair_set.append([sim[i][j], i, j])
    sorted(pair_set, key=(lambda x: x[0]), reverse=True)
    seed_num = int(0.05*min(sim.shape[0], sim.shape[1]))
    seed = [[pair_set[i][1], pair_set[i][2]] for i in range(seed_num)]
    seed1 = []
    seed2 = []
    seed_file_1 = open(config.seed_file1 + str(ratio) + ".pkl", 'wb')
    seed_file_2 = open(config.seed_file2 + str(ratio) + ".pkl", 'wb')

    for pair in seed:
        seed1.append(pair[0])
        seed2.append(pair[1])
    pickle.dump(seed1, seed_file_1)
    pickle.dump(seed2, seed_file_2)
    seed_file_1.close()
    seed_file_2.close()

"""
calculate shortest path distance embedding
"""
def shortest_path_emd(ratio):

    seed_file_1 = open(config.seed_file1 + str(ratio) + ".pkl", 'rb')
    seed_file_2 = open(config.seed_file2 + str(ratio) + ".pkl", 'rb')
    seed1 = pickle.load(seed_file_1)
    seed2 = pickle.load(seed_file_2)
    seed_file_1.close()
    seed_file_2.close()
    network1 = nx.Graph()
    network2 = nx.Graph()
    f1 = open(config.norm_g1_file, 'r')
    f2 = open(config.norm_g2_file, 'r')
    for line in f1.readlines():
        edge = line.strip('\n').split()
        network1.add_edge(int(edge[0]), int(edge[1]))
    for line in f2.readlines():
        edge = line.strip('\n').split()
        network2.add_edge(int(edge[0]), int(edge[1]))
    f1.close()
    f2.close()
    g1_dis = []
    g2_dis = []
    for anchor in seed1:
        p = [0 for i in range(len(network1.nodes))]
        anchor_dis = nx.shortest_path_length(network1, source=anchor)
        for key in anchor_dis.keys():
            p[key] = anchor_dis[key]
        g1_dis.append(p)
    for anchor in seed2:
        p = [0 for i in range(len(network2.nodes))]
        anchor_dis = nx.shortest_path_length(network2, source=anchor)
        for key in anchor_dis.keys():
            p[key] = anchor_dis[key]
        g2_dis.append(p)
    g1_rwr_emd = np.array(g1_dis, dtype=float).T
    g2_rwr_emd = np.array(g2_dis, dtype=float).T
    rwr_emd_1_file = open(config.rwr1_emd + str(ratio) + ".pkl", 'wb')
    rwr_emd_2_file = open(config.rwr2_emd + str(ratio) + ".pkl", 'wb')
    pickle.dump(g1_rwr_emd, rwr_emd_1_file)
    pickle.dump(g2_rwr_emd, rwr_emd_2_file)
    rwr_emd_1_file.close()
    rwr_emd_2_file.close()


"""
calculate RWR embedding 
"""
def rwr_emd(ratio):

    rwr1 = RWR()
    rwr2 = RWR()
    rwr1.read_graph(config.norm_g1_file, "undirected")
    rwr2.read_graph(config.norm_g2_file, "undirected")
    seed_file_1 = open(config.seed_file1 + str(ratio) + ".pkl", 'rb')
    seed_file_2 = open(config.seed_file2 + str(ratio) + ".pkl", 'rb')
    seed1 = pickle.load(seed_file_1)
    seed2 = pickle.load(seed_file_2)
    seed_file_1.close()
    seed_file_2.close()
    g1_rwr = []
    g2_rwr = []
    for anchor in seed1:
        emd = rwr1.compute(anchor)
        g1_rwr.append(list(emd))
    for anchor in seed2:
        emd = rwr2.compute(anchor)
        g2_rwr.append(list(emd))
    g1_rwr_emd = np.array(g1_rwr, dtype=float).T
    g2_rwr_emd = np.array(g2_rwr, dtype=float).T
    rwr_emd_1_file = open(config.rwr1_emd + str(ratio) + ".pkl", 'wb')
    rwr_emd_2_file = open(config.rwr2_emd + str(ratio) + ".pkl", 'wb')
    pickle.dump(g1_rwr_emd, rwr_emd_1_file)
    pickle.dump(g2_rwr_emd, rwr_emd_2_file)
    rwr_emd_1_file.close()
    rwr_emd_2_file.close()
    seed_file_1.close()
    seed_file_2.close()


"""
construct attribute for cora dataset
"""
def build_gcn_data_cora():

    norm_g1_file = open(config.norm_g1_file, 'r')
    norm_g2_file = open(config.norm_g2_file, 'r')
    g1_edge = [[], []]
    g2_edge = [[], []]
    g1_node = set()
    g2_node = set()
    for line in norm_g1_file.readlines():
        edge = line.strip('\n').split()
        g1_node.add(int(edge[0]))
        g1_node.add(int(edge[1]))
        g1_edge[0].append(int(edge[0]))
        g1_edge[1].append(int(edge[1]))
    for line in norm_g2_file.readlines():
        edge = line.strip('\n').split()
        g2_node.add(int(edge[0]))
        g2_node.add(int(edge[1]))
        g2_edge[0].append(int(edge[0]))
        g2_edge[1].append(int(edge[1]))
    norm_g1_file.close()
    norm_g2_file.close()
    g1_edge = np.array(g1_edge, dtype=np.long)
    g2_edge = np.array(g2_edge, dtype=np.long)
    data = np.load('%s.npz' % config.numpy_file)
    g1_feat, g2_feat = data['x1'], data['x2']
    gcn_data_file = open(config.gcn_data, 'wb')
    pickle.dump([g1_feat, g1_edge, g2_feat, g2_edge], gcn_data_file)
    gcn_data_file.close()

"""
construct one-hot attribute for plain network for comparison 
"""
def build_gcn_data():

    norm_g1_file = open(config.norm_g1_file, 'r')
    norm_g2_file = open(config.norm_g2_file, 'r')
    g1_edge = [[], []]
    g2_edge = [[], []]
    g1_node = set()
    g2_node = set()
    for line in norm_g1_file.readlines():
        edge = line.strip('\n').split()
        g1_node.add(int(edge[0]))
        g1_node.add(int(edge[1]))
        g1_edge[0].append(int(edge[0]))
        g1_edge[1].append(int(edge[1]))
    for line in norm_g2_file.readlines():
        edge = line.strip('\n').split()
        g2_node.add(int(edge[0]))
        g2_node.add(int(edge[1]))
        g2_edge[0].append(int(edge[0]))
        g2_edge[1].append(int(edge[1]))
    norm_g1_file.close()
    norm_g2_file.close()
    g1_edge = np.array(g1_edge, dtype=np.long)
    g2_edge = np.array(g2_edge, dtype=np.long)
    g1_feat = np.zeros([len(g1_node), len(g1_node)])
    for i in range(g1_feat.shape[0]):
        g1_feat[i][i] = 1
    g2_feat = np.zeros([len(g2_node), len(g2_node)])
    for i in range(g2_feat.shape[0]):
        g2_feat[i][i] = 1
    gcn_data_file = open(config.gcn_data, 'wb')
    pickle.dump([g1_feat, g1_edge, g2_feat, g2_edge], gcn_data_file)

"""
random negative sampling
"""
def get_rand_neg(out1, out2, k, anchor1, anchor2):
    neg1 = []
    neg2 = []
    t = len(anchor1)
    G1_vec = np.array(out1)
    G2_vec = np.array(out2)
    G1_nodes = [i for i in range(G1_vec.shape[0])]
    G2_nodes = [i for i in range(G2_vec.shape[0])]
    for i in range(t):
        rand_sample_1 = random.sample(G2_nodes, k)
        rand_sample_2 = random.sample(G1_nodes, k)
        neg1.append(rand_sample_1)
        neg2.append(rand_sample_2)
    neg1 = np.array(neg1)
    neg1 = neg1.reshape((t * k,))
    neg2 = np.array(neg2)
    neg2 = neg2.reshape((t * k,))
    anchor1 = np.repeat(anchor1, k)
    anchor2 = np.repeat(anchor2, k)
    return anchor1, anchor2, neg1, neg2

"""
advanced negative sampling 
"""
def get_neg(out1, out2, k, anchor1, anchor2):
    neg1 = []
    neg2 = []
    t = len(anchor1)
    anchor1_vec = np.array(out1[anchor1])
    anchor2_vec = np.array(out2[anchor2])
    G1_vec = np.array(out1)
    G2_vec = np.array(out2)
    sim1 = sp.distance.cdist(anchor1_vec, G2_vec, metric='cityblock')
    for i in range(t):
        rank = sim1[i, :].argsort()
        neg1.append(rank[0:k])
    neg1 = np.array(neg1)
    neg1 = neg1.reshape((t * k,))
    sim2 = sp.distance.cdist(anchor2_vec, G1_vec, metric='cityblock')
    for i in range(t):
        rank = sim2[i, :].argsort()
        neg2.append(rank[0:k])
    anchor1 = np.repeat(anchor1, k)
    anchor2 = np.repeat(anchor2, k)
    neg2 = np.array(neg2)
    neg2 = neg2.reshape((t * k,))
    return anchor1, anchor2, neg1, neg2

"""
evaluation
"""
def get_hits(out1, out2, test_pair, top_k=(1, 5, 10, 30, 50, 100)):
    test_L = [pair[0] for pair in test_pair]
    test_R = [pair[1] for pair in test_pair]
    Lvec = np.array(out1[test_L])
    Rvec = np.array(out2[test_R])
    Lmap = {}
    Rmap = {}
    for pair in test_pair:
        [e1, e2] = pair
        Lmap[e1] = e2
        Rmap[e2] = e1
    sim1 = sp.distance.cdist(Lvec, out2, metric='cityblock')
    sim2 = sp.distance.cdist(Rvec, out1, metric='cityblock')
    top_lr = [0] * len(top_k)
    L_mrr = 0
    R_mrr = 0
    for i in range(Lvec.shape[0]):
        rank1 = sim1[i, :].argsort()
        rank_index1 = np.where(rank1 == Lmap[test_pair[i][0]])[0][0]
        L_mrr += 1/(rank_index1 + 1)
        for j in range(len(top_k)):
            if rank_index1 < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * (len(top_k))
    for i in range(Rvec.shape[0]):
        rank2 = sim2[i, :].argsort()
        rank_index2 = np.where(rank2 == Rmap[test_pair[i][1]])[0][0]
        R_mrr += 1/(rank_index2 + 1)
        for j in range(len(top_k)):
            if rank_index2 < top_k[j]:
                top_rl[j] += 1
    L_mrr = L_mrr/len(test_pair)
    R_mrr = R_mrr/len(test_pair)
    result = []
    for i in range(len(top_lr)):
        result.append(top_lr[i] / len(test_pair) * 100)
    result.append(L_mrr)
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print("MRR is %.2f%%" % (L_mrr * 100))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    print("MRR is %.2f%%" % (R_mrr * 100))
    return result

"""
the parameter norm refers to clean the ori_data into norm_data 
"""
def preprocess(ratio, used_rwr=True, norm=False):
    if ratio != 0:
        if config.data == "F2T":
            if norm:
                ori2norm_F2T()
            split_data(ratio)
            build_gcn_data()
            if used_rwr:
               rwr_emd(ratio)
            else:
                shortest_path_emd(ratio)

        else:
            if config.data == "D2A":
                if norm:
                    ori2norm_D2A()
                if used_rwr:
                    rwr_emd(ratio)
                else:
                    shortest_path_emd(ratio)
            if config.data == "Cora":
                if norm:
                    ori2norm_Cora()
                split_data(ratio)
                build_gcn_data_cora()
                rwr_emd(ratio)
            if config.data == "DBLP":
                if norm:
                    ori2norm_Cora()
                split_data(ratio)
                build_gcn_data_cora()
                rwr_emd(ratio)
    else:
        if config.data == "Cora":
            if norm:
                ori2norm_Cora()
            split_data(ratio)
            get_candi_seed()
            build_gcn_data_cora()
            rwr_emd(ratio)
        if config.data == "DBLP":
            if norm:
                ori2norm_Cora()
            split_data(ratio)
            get_candi_seed()
            build_gcn_data_cora()
            rwr_emd(ratio)

























