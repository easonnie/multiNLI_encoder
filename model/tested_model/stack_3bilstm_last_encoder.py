import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import numpy as np
import torch_util
from tqdm import tqdm
from model.baseModel import model_eval
import util.save_tool as save_tool
import os
from datetime import datetime

import util.data_loader as data_loader
import config


class StackBiLSTMMaxout(nn.Module):
    def __init__(self, h_size=[512, 1024, 2048], v_size=10, d=300, mlp_d=1600, dropout_r=0.1, max_l=60):
        super(StackBiLSTMMaxout, self).__init__()
        self.Embd = nn.Embedding(v_size, d)

        self.lstm = nn.LSTM(input_size=d, hidden_size=h_size[0],
                            num_layers=1, bidirectional=True)

        self.lstm_1 = nn.LSTM(input_size=(d + h_size[0] * 2), hidden_size=h_size[1],
                              num_layers=1, bidirectional=True)

        self.lstm_2 = nn.LSTM(input_size=(d + (h_size[0] + h_size[1]) * 2), hidden_size=h_size[2],
                              num_layers=1, bidirectional=True)

        self.max_l = max_l
        self.h_size = h_size

        self.mlp_1 = nn.Linear(h_size[2] * 2 * 4, mlp_d)
        self.mlp_2 = nn.Linear(mlp_d, mlp_d)
        self.sm = nn.Linear(mlp_d, 3)

        self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(dropout_r),
                                          self.mlp_2, nn.ReLU(), nn.Dropout(dropout_r),
                                          self.sm])

    def display(self):
        for param in self.parameters():
            print(param.data.size())

    def forward(self, s1, l1, s2, l2):
        if self.max_l:
            l1 = l1.clamp(max=self.max_l)
            l2 = l2.clamp(max=self.max_l)
            if s1.size(0) > self.max_l:
                s1 = s1[:self.max_l, :]
            if s2.size(0) > self.max_l:
                s2 = s2[:self.max_l, :]

        p_s1 = self.Embd(s1)
        p_s2 = self.Embd(s2)

        s1_layer1_out = torch_util.auto_rnn_bilstm(self.lstm, p_s1, l1)
        s2_layer1_out = torch_util.auto_rnn_bilstm(self.lstm, p_s2, l2)

        # Length truncate
        len1 = s1_layer1_out.size(0)
        len2 = s2_layer1_out.size(0)
        p_s1 = p_s1[:len1, :, :] # [T, B, D]
        p_s2 = p_s2[:len2, :, :] # [T, B, D]

        # Using residual connection
        s1_layer2_in = torch.cat([p_s1, s1_layer1_out], dim=2)
        s2_layer2_in = torch.cat([p_s2, s2_layer1_out], dim=2)

        s1_layer2_out = torch_util.auto_rnn_bilstm(self.lstm_1, s1_layer2_in, l1)
        s2_layer2_out = torch_util.auto_rnn_bilstm(self.lstm_1, s2_layer2_in, l2)

        s1_layer3_in = torch.cat([p_s1, s1_layer1_out, s1_layer2_out], dim=2)
        s2_layer3_in = torch.cat([p_s2, s2_layer1_out, s2_layer2_out], dim=2)

        s1_layer3_out = torch_util.auto_rnn_bilstm(self.lstm_2, s1_layer3_in, l1)
        s2_layer3_out = torch_util.auto_rnn_bilstm(self.lstm_2, s2_layer3_in, l2)

        s1_layer3_maxout = torch_util.max_along_time(s1_layer3_out, l1)
        s2_layer3_maxout = torch_util.max_along_time(s2_layer3_out, l2)

        # Only use the last layer
        features = torch.cat([s1_layer3_maxout, s2_layer3_maxout,
                              torch.abs(s1_layer3_maxout - s2_layer3_maxout),
                              s1_layer3_maxout * s2_layer3_maxout],
                             dim=1)

        out = self.classifier(features)
        return out


def train(combined_set=False):
    torch.manual_seed(6)

    snli_d, mnli_d, embd = data_loader.load_data_sm(
        config.DATA_ROOT, config.EMBD_FILE, reseversed=False, batch_sizes=(32, 200, 200, 30, 30), device=0)

    s_train, s_dev, s_test = snli_d
    m_train, m_dev_m, m_dev_um, m_test_m, m_test_um = mnli_d

    s_train.repeat = False
    m_train.repeat = False

    model = StackBiLSTMMaxout()
    model.Embd.weight.data = embd
    model.display()

    if torch.cuda.is_available():
        embd.cuda()
        model.cuda()

    start_lr = 2e-4

    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    criterion = nn.CrossEntropyLoss()

    date_now = datetime.now().strftime("%m-%d-%H:%M:%S")
    name = '[512,1024,2048]-3stack-bilstm-last_maxout'
    file_path = save_tool.gen_prefix(name, date_now)

    """
    Attention:!!!
        Modify this to save to log file.
    """
    message = "w(300) -> 512 bilstm -> h1(2048)\n" + \
              "[w(300), h1(2048)] -> 2048 bilstm -> h2(4096) -> maxout -> h2(4096) //Using high way\n" + \
              "[h2,h2,abs(h1-h2),h1*h2](4096 * 4) -> 1200-mlp -> 1600-mlp -> 3-sm\n"

    save_tool.logging2file(file_path, 'code', None, __file__)
    save_tool.logging2file(file_path, 'message', message, __file__)


    iterations = 0

    best_m_dev = -1
    best_um_dev = -1

    param_file_prefix = "{}/{}".format(file_path, "saved_params")
    if not os.path.exists(os.path.join(config.ROOT_DIR, param_file_prefix)):
        os.mkdir(os.path.join(config.ROOT_DIR, param_file_prefix))

    for i in range(6):
        s_train.init_epoch()
        m_train.init_epoch()

        if not combined_set:
            train_iter, dev_iter, test_iter = s_train, s_dev, s_test
            train_iter.repeat = False
            print(len(train_iter))
        else:
            train_iter = data_loader.combine_two_set(s_train, m_train, rate=[0.15, 1], seed=i)
            dev_iter, test_iter = s_dev, s_test

        best_dev = -1
        best_test = -1
        start_perf = model_eval(model, dev_iter, criterion)
        i_decay = i // 2
        lr = start_lr / (2 ** i_decay)

        epoch_start_info = "epoch:{}, learning_rate:{}, start_performance:{}/{}\n".format(i, lr, *start_perf)
        print(epoch_start_info)
        save_tool.logging2file(file_path, 'log', epoch_start_info)

        if i != 0:
            SAVE_PATH = os.path.join(config.ROOT_DIR, file_path, 'm_{}'.format(i - 1))
            model.load_state_dict(torch.load(SAVE_PATH))

        for batch_idx, batch in tqdm(enumerate(train_iter)):
            iterations += 1
            model.train()

            s1, s1_l = batch.premise
            s2, s2_l = batch.hypothesis
            y = batch.label - 1

            out = model(s1, (s1_l - 1), s2, (s2_l - 1))
            loss = criterion(out, y)

            optimizer.zero_grad()

            for pg in optimizer.param_groups:
                pg['lr'] = lr

            loss.backward()
            optimizer.step()

            if i == 0 or i == 1:
                mod = 6000
            else:
                mod = 100

            if (1 + batch_idx) % mod == 0:
                dev_score, dev_loss = model_eval(model, dev_iter, criterion)
                test_score, test_loss = -1, -1
                print('SNLI:{}/{}'.format(dev_score, dev_loss), end=' ')

                model.max_l = 150
                mdm_score, mdm_loss = model_eval(model, m_dev_m, criterion)
                mdum_score, mdum_loss = model_eval(model, m_dev_um, criterion)

                print(' MNLI_M:{}/{}'.format(mdm_score, mdm_loss), end=' ')
                print(' MNLI_UM:{}/{}'.format(mdum_score, mdum_loss))
                model.max_l = 60

                now = datetime.now().strftime("%m-%d-%H:%M:%S")
                log_info = "{}\t{}\tdev:{}/{}\ttest:{}/{}\t{}\n".format(i, iterations, dev_score, dev_loss, test_score, test_loss, now)
                save_tool.logging2file(file_path, "log", log_info)
                log_info_mnli = "dev_m:{}/{} um:{}/{}\n".format(mdm_score, mdm_loss, mdum_score, mdum_loss)
                save_tool.logging2file(file_path, "log", log_info_mnli)

                saved = False
                if best_m_dev < mdm_score:
                    best_m_dev = mdm_score
                    save_path = os.path.join(config.ROOT_DIR, param_file_prefix,
                                             'e({})_m_m({})_um({})'.format(i, mdm_score, mdum_score))
                    torch.save(model.state_dict(), save_path)
                    saved = True

                if best_um_dev < mdum_score:
                    best_um_dev = mdum_score
                    save_path = os.path.join(config.ROOT_DIR, param_file_prefix,
                                             'e({})_m_m({})_um({})'.format(i, mdm_score, mdum_score))
                    if not saved:
                        torch.save(model.state_dict(), save_path)

        SAVE_PATH = os.path.join(config.ROOT_DIR, file_path, 'm_{}'.format(i))
        torch.save(model.state_dict(), SAVE_PATH)
        print(best_test)


def evaluation():
    torch.manual_seed(6)

    snli_d, mnli_d, embd = data_loader.load_data_sm(
        config.DATA_ROOT, config.EMBD_FILE, reseversed=False, batch_sizes=(32, 32, 32, 32, 32), device=0)

    # s_train, s_dev, s_test = snli_d
    m_train, m_dev_m, m_dev_um, m_test_m, m_test_um = mnli_d

    m_test_um.shuffle = False
    m_test_m.shuffle = False
    m_test_um.sort = False
    m_test_m.sort = False

    model = StackBiLSTMMaxout()
    model.Embd.weight.data = embd
    model.display()

    if torch.cuda.is_available():
        embd.cuda()
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    # /home/easonnie/Projects/ssu/
    # /home/easonnie/Projects/ssu/
    # /home/easonnie/Projects/ssu/saved_model/06-12-10:33:16_[1024,2048]-stack-bilstm-last_maxout/saved_params/
    # /home/easonnie/Projects/ssu/saved_model/06-12-10:33:16_[1024,2048]-stack-bilstm-last_maxout/saved_params/
    # /home/easonnie/Projects/ssu/saved_model/06-13-02:21:31_[512,1024,2048]-3stack-bilstm-last_maxout/saved_params/e(2)_m_m(74.1110545084055)_um(74.55248169243288)
    # /home/easonnie/Projects/ssu/saved_model/06-13-02:21:31_[512,1024,2048]-3stack-bilstm-last_maxout/saved_params/e(2)_m_m(73.84615384615384)_um(75.0)
    # /home/easonnie/Projects/ssu/saved_model/06-14-01:31:38_[512,1024,2048]-3stack-bilstm+cnn-last_maxout/saved_params/e(2)_m_m(74.01935812531839)_um(74.11513425549226)

    file_path = "saved_model/06-14-01:31:38_[512,1024,2048]-3stack-bilstm+cnn-last_maxout/saved_params/e(2)_m_m(74.01935812531839)_um(74.11513425549226)"
    SAVE_PATH = os.path.join(config.ROOT_DIR, file_path)
    model.load_state_dict(torch.load(SAVE_PATH))

    m_pred = model_eval(model, m_test_m, criterion, pred=True)
    um_pred = model_eval(model, m_test_um, criterion, pred=True)

    model.max_l = 150
    print(um_pred)
    print(m_pred)

    with open('./sub_um.csv', 'w+') as f:
        index = ['entailment', 'contradiction', 'neutral']
        f.write("pairID,gold_label\n")
        for i, k in enumerate(um_pred):
            f.write(str(i) + "," + index[k] + "\n")

    with open('./sub_m.csv', 'w+') as f:
        index = ['entailment', 'contradiction', 'neutral']
        f.write("pairID,gold_label\n")
        for j, k in enumerate(m_pred):
            f.write(str(j + 9847) + "," + index[k] + "\n")


if __name__ == '__main__':
    train(True)
    # evaluation()
    # train_and_fine_select(True)