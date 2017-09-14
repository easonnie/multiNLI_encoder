import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import numpy as np
import torch_util
from tqdm import tqdm

import util.data_loader as data_loader
import config


class BiLSTMMaxout(nn.Module):
    def __init__(self, h_size=512, v_size=10, d=300, mlp_d=600, dropout_r=0.1):
        super(BiLSTMMaxout, self).__init__()
        self.Embd = nn.Embedding(v_size, d)
        self.lstm = nn.LSTM(input_size=d, hidden_size=h_size,
                            num_layers=1, bidirectional=True)
        self.h_size = h_size

        # self.feature_bn = nn.BatchNorm1d(h_size * 3 * 2)

        self.mlp_1 = nn.Linear(h_size * 4 * 2, mlp_d)
        self.mlp_2 = nn.Linear(mlp_d, mlp_d)
        self.sm = nn.Linear(mlp_d, 3)

        # self.classifier = nn.Sequential(*[self.feature_bn,
        #                                   self.mlp_1, nn.ReLU(), nn.Dropout(dropout_r),
        #                                   self.mlp_2, nn.ReLU(), nn.Dropout(dropout_r),
        #                                   self.sm])

        self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(dropout_r),
                                          self.mlp_2, nn.ReLU(), nn.Dropout(dropout_r),
                                          self.sm])

    def display(self):
        for param in self.parameters():
            # print(param.creator())
            print(param.data.size())

    def forward(self, s1, l1, s2, l2):
        p_s1 = self.Embd(s1)
        p_s2 = self.Embd(s2)

        s1_a_out = torch_util.auto_rnn_bilstm(self.lstm, p_s1, l1)
        s2_a_out = torch_util.auto_rnn_bilstm(self.lstm, p_s2, l2)

        s1_max_out = torch_util.max_along_time(s1_a_out, l1)
        s2_max_out = torch_util.max_along_time(s2_a_out, l2)

        features = torch.cat([s1_max_out, s2_max_out, torch.abs(s1_max_out - s2_max_out), s1_max_out * s2_max_out], dim=1)

        out = self.classifier(features)
        return out


def model_eval(model, data_iter, criterion, pred=False):
    model.eval()
    data_iter.init_epoch()
    n_correct = loss = 0
    totoal_size = 0

    if not pred:
        for batch_idx, batch in enumerate(data_iter):

            s1, s1_l = batch.premise
            s2, s2_l = batch.hypothesis
            y = batch.label.data - 1

            pred = model(s1, s1_l - 1, s2, s2_l - 1)
            n_correct += (torch.max(pred, 1)[1].view(batch.label.size()).data == y).sum()

            loss += criterion(pred, batch.label - 1).data[0] * batch.batch_size
            totoal_size += batch.batch_size

        avg_acc = 100. * n_correct / totoal_size
        avg_loss = loss / totoal_size

        return avg_acc, avg_loss
    else:
        pred_list = []
        for batch_idx, batch in enumerate(data_iter):

            s1, s1_l = batch.premise
            s2, s2_l = batch.hypothesis

            pred = model(s1, s1_l - 1, s2, s2_l - 1)
            pred_list.append(torch.max(pred, 1)[1].view(batch.label.size()).data)

        return torch.cat(pred_list, dim=0)


def model_output_vector(model, data_iter):
    model.eval()
    data_iter.init_epoch()

    p_list = []
    h_list = []

    for batch_idx, batch in enumerate(data_iter):
        s1, s1_l = batch.premise
        s2, s2_l = batch.hypothesis

        p_vector = model.encoding(s1, s1_l - 1)
        h_vector = model.encoding(s2, s2_l - 1)

        p_list.append(p_vector.data.numpy())
        h_list.append(h_vector.data.numpy())
        # print(h_vector.data.numpy().shape)

    # print(p_list)
    # print(h_list)

    p_whole = np.concatenate(p_list, axis=0)
    h_whole = np.concatenate(h_list, axis=0)

    return p_whole, h_whole


def train_BiLSTMMaxout():
    torch.manual_seed(6)

    train_iter, dev_iter, test_iter, embd = data_loader.load_data(
        config.DATA_ROOT, config.EMBD_FILE, reseversed=False, batch_sizes=(32, 500, 500), device=0)

    model = BiLSTMMaxout()
    embd.cuda()
    model.Embd.weight.data = embd
    # model.display()

    SAVE_PATH = config.ROOT_DIR + '/saved_model/bilstmMax_512_2/m'
    model.load_state_dict(torch.load(SAVE_PATH))

    if torch.cuda.is_available():
        model.cuda()
    # print(model.Embd.weight)
    # # resume

    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    iterations = 0

    print(model_eval(model, test_iter, criterion))

    train_iter.repeat = False
    print(len(train_iter))

    best_dev = -1
    best_test = -1

    for i in range(1):
        for batch_idx, batch in tqdm(enumerate(train_iter)):
            train_iter.init_epoch()
            iterations += 1
            model.train()

            s1, s1_l = batch.premise
            s2, s2_l = batch.hypothesis
            y = batch.label - 1

            out = model(s1, (s1_l - 1), s2, (s2_l - 1))
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iterations % 100 == 0:
                dev_score, dev_loss = model_eval(model, dev_iter, criterion)
                print(dev_score, dev_loss)
                if dev_score >= best_dev:
                    best_dev = dev_score
                    test_score, test_loss = model_eval(model, test_iter, criterion)
                    if test_score > best_test:
                        best_test = test_score
                        print('Test:', best_test)

    SAVE_PATH = config.ROOT_DIR + '/saved_model/bilstmMax_512_3/m'
    torch.save(model.state_dict(), SAVE_PATH)
    print(best_test)


if __name__ == '__main__':
    train_BiLSTMMaxout()