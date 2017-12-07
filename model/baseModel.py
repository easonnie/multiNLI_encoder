import torch
import torch.nn as nn
import torch_util


class BiLSTMMaxout(nn.Module):
    def __init__(self, h_size=512, v_size=10, d=300, mlp_d=600, dropout_r=0.1):
        super(BiLSTMMaxout, self).__init__()
        self.Embd = nn.Embedding(v_size, d)
        self.lstm = nn.LSTM(input_size=d, hidden_size=h_size,
                            num_layers=1, bidirectional=True)
        self.h_size = h_size

        self.mlp_1 = nn.Linear(h_size * 4 * 2, mlp_d)
        self.mlp_2 = nn.Linear(mlp_d, mlp_d)
        self.sm = nn.Linear(mlp_d, 3)

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


if __name__ == '__main__':
    pass