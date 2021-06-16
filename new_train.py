from model import ranking_loss_L1, BRIGHT_A, BRIGHT_U, BRIGHT_gcn
import bright_utils
import pickle
import torch
from torch_geometric.data import Data
import config


def train(ratio, epoch_num, lr, dim, k, gamma, use_rwr=True, use_gcn=True):

    gcn_data_file = open(config.gcn_data, 'rb')
    [g1_feat, g1_edge, g2_feat, g2_edge] = pickle.load(gcn_data_file)
    gcn_data_file.close()
    rwr_emd_1_file = open(config.rwr1_emd + str(ratio) + ".pkl", 'rb')
    rwr_emd_2_file = open(config.rwr2_emd + str(ratio) + ".pkl", 'rb')
    rwr_emd_1 = pickle.load(rwr_emd_1_file)
    rwr_emd_2 = pickle.load(rwr_emd_2_file)
    rwr_emd_1_file.close()
    rwr_emd_2_file.close()
    seed_file_1 = open(config.seed_file1 + str(ratio) + ".pkl", 'rb')
    seed_file_2 = open(config.seed_file2 + str(ratio) + ".pkl", 'rb')
    seed1 = pickle.load(seed_file_1)
    seed2 = pickle.load(seed_file_2)
    seed_file_1.close()
    seed_file_2.close()
    test_file = open(config.test_file + str(ratio) + ".pkl", 'rb')
    test = pickle.load(test_file)
    test_file.close()
    rwr_dim = rwr_emd_1.shape[1]
    g1_feat_dim = g1_feat.shape[1]
    g2_feat_dim = g2_feat.shape[1]
    g1_edge_index = torch.tensor(g1_edge, dtype=torch.long)
    g2_edge_index = torch.tensor(g2_edge, dtype=torch.long)
    g1_data_x = torch.tensor(g1_feat, dtype=torch.float)
    g2_data_x = torch.tensor(g2_feat, dtype=torch.float)
    g1_data = Data(x=g1_data_x, edge_index=g1_edge_index)
    g2_data = Data(x=g2_data_x, edge_index=g2_edge_index)
    rwr_emd_1 = torch.tensor(rwr_emd_1, dtype=torch.float)
    rwr_emd_2 = torch.tensor(rwr_emd_2, dtype=torch.float)
    device = torch.device('cuda:0')
    criterion = ranking_loss_L1().to(device)
    if use_gcn and use_rwr:
        model = BRIGHT_A(dim, rwr_dim, g1_feat_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        g1_data_gpu = g1_data.to(device)
        g2_data_gpu = g2_data.to(device)
        rwr_emd_1_gpu = rwr_emd_1.to(device)
        rwr_emd_2_gpu = rwr_emd_2.to(device)
        print("Start Training......")
        for epoch in range(epoch_num):
            out1, out2 = model(rwr_emd_1_gpu, rwr_emd_2_gpu, g1_data_gpu, g2_data_gpu)
            np_out1 = out1.detach().cpu()
            np_out2 = out2.detach().cpu()
            anchor1, anchor2, neg1, neg2 = bright_utils.get_neg(np_out1, np_out2, k, seed1, seed2)
            optimizer.zero_grad()
            out1, out2 = model(rwr_emd_1_gpu, rwr_emd_2_gpu, g1_data_gpu, g2_data_gpu)
            loss = criterion(out1, out2, anchor1, anchor2, neg1, neg2, gamma)
            loss.backward()
            optimizer.step()
            print("Epoch: ", epoch)
            print("loss: ", loss.cpu().detach().numpy())
            if epoch % 5 == 0:
                result = bright_utils.get_hits(np_out1, np_out2, test)
                print(result)
    else:
        if use_rwr:
            model = BRIGHT_U(dim, rwr_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            rwr_emd_1_gpu = rwr_emd_1.to(device)
            rwr_emd_2_gpu = rwr_emd_2.to(device)
            print("Start Training......")
            for epoch in range(epoch_num):
                out1, out2 = model(rwr_emd_1_gpu, rwr_emd_2_gpu)
                np_out1 = out1.detach().cpu()
                np_out2 = out2.detach().cpu()
                anchor1, anchor2, neg1, neg2 = bright_utils.get_neg(np_out1, np_out2, k, seed1, seed2)
                optimizer.zero_grad()
                out1, out2 = model(rwr_emd_1_gpu, rwr_emd_2_gpu)
                loss = criterion(out1, out2, anchor1, anchor2, neg1, neg2, gamma)
                loss.backward()
                optimizer.step()
                print("Epoch: ", epoch)
                print("loss: ", loss.cpu().detach().numpy())
                if epoch % 5 == 0:
                    result = bright_utils.get_hits(np_out1, np_out2, test)
                    print(result)
        else:
            model = BRIGHT_gcn(dim, g1_feat_dim, g2_feat_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            g1_data_gpu = g1_data.to(device)
            g2_data_gpu = g2_data.to(device)
            print("Start Training......")
            for epoch in range(epoch_num):
                out1, out2 = model(g1_data_gpu, g2_data_gpu)
                np_out1 = out1.detach().cpu()
                np_out2 = out2.detach().cpu()
                anchor1, anchor2, neg1, neg2 = bright_utils.get_neg(np_out1, np_out2, k, seed1, seed2)
                optimizer.zero_grad()
                out1, out2 = model(g1_data_gpu, g2_data_gpu)
                loss = criterion(out1, out2, anchor1, anchor2, neg1, neg2, gamma)
                loss.backward()
                optimizer.step()
                print("Epoch: ", epoch)
                print("loss: ", loss.cpu().detach().numpy())
                if epoch % 5 == 0:
                    result = bright_utils.get_hits(np_out1, np_out2, test)
                    print(result)








