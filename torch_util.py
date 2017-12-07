import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def pad(t, length):
    if length == t.size(0):
        return t
    else:
        return torch.cat([t, Variable(t.data.new(length - t.size(0), *t.size()[1:]).zero_())])


def pack_list_sequence(inputs, l):
    batch_list = []
    max_l = max(list(l))
    batch_size = len(inputs)

    for b_i in range(batch_size):
        batch_list.append(pad(inputs[b_i], max_l))
    pack_batch_list = torch.stack(batch_list, dim=1)
    return pack_batch_list


def pack_for_rnn_seq(inputs, lengths):
    """
    :param inputs: [T * B * D] 
    :param lengths:  [B]
    :return: 
    """
    _, sorted_indices = lengths.sort()
    '''
        Reverse to decreasing order
    '''
    r_index = reversed(list(sorted_indices))

    s_inputs_list = []
    lengths_list = []
    reverse_indices = np.zeros(lengths.size(0), dtype=np.int64)

    for j, i in enumerate(r_index):
        s_inputs_list.append(inputs[:, i, :].unsqueeze(1))
        lengths_list.append(lengths[i])
        reverse_indices[i] = j

    reverse_indices = list(reverse_indices)

    s_inputs = torch.cat(s_inputs_list, 1)
    packed_seq = nn.utils.rnn.pack_padded_sequence(s_inputs, lengths_list)

    return packed_seq, reverse_indices


def unpack_from_rnn_seq(packed_seq, reverse_indices):
    unpacked_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_seq)
    s_inputs_list = []

    for i in reverse_indices:
        s_inputs_list.append(unpacked_seq[:, i, :].unsqueeze(1))
    return torch.cat(s_inputs_list, 1)


def auto_rnn_bilstm(lstm: nn.LSTM, seqs, lengths):

    batch_size = seqs.size(1)

    state_shape = lstm.num_layers * 2, batch_size, lstm.hidden_size

    h0 = c0 = Variable(seqs.data.new(*state_shape).zero_())

    packed_pinputs, r_index = pack_for_rnn_seq(seqs, lengths)

    output, (hn, cn) = lstm(packed_pinputs, (h0, c0))

    output = unpack_from_rnn_seq(output, r_index)

    return output


def auto_rnn_bigru(gru: nn.GRU, seqs, lengths):

    batch_size = seqs.size(1)

    state_shape = gru.num_layers * 2, batch_size, gru.hidden_size

    h0 = Variable(seqs.data.new(*state_shape).zero_())

    packed_pinputs, r_index = pack_for_rnn_seq(seqs, lengths)

    output, hn = gru(packed_pinputs, h0)

    output = unpack_from_rnn_seq(output, r_index)

    return output


def select_last(inputs, lengths, hidden_size):
    """
    :param inputs: [T * B * D] D = 2 * hidden_size
    :param lengths: [B]
    :param hidden_size: dimension 
    :return:  [B * D]
    """
    batch_size = inputs.size(1)
    batch_out_list = []
    for b in range(batch_size):
        batch_out_list.append(torch.cat((inputs[lengths[b] - 1, b, :hidden_size],
                                         inputs[0, b, hidden_size:])
                                        )
                              )

    out = torch.stack(batch_out_list)
    return out


def channel_weighted_sum(s, w, l, sharpen=None):
    batch_size = w.size(1)
    result_list = []
    for b_i in range(batch_size):
        if sharpen:
            b_w = w[:l[b_i], b_i, :] * sharpen
        else:
            b_w = w[:l[b_i], b_i, :]
        b_s = s[:l[b_i], b_i, :] # T, D
        soft_b_w = F.softmax(b_w.transpose(0, 1)).transpose(0, 1)
        # print(soft_b_w)
        # print('soft:', )
        # print(soft_b_w)
        result_list.append(torch.sum(soft_b_w * b_s, dim=0)) # [T, D] -> [1, D]
    return torch.cat(result_list, dim=0)


def pack_to_matching_matrix(s1, s2, cat_only=[False, False]):
    t1 = s1.size(0)
    t2 = s2.size(0)
    batch_size = s1.size(1)
    d = s1.size(2)

    expanded_p_s1 = s1.expand(t2, t1, batch_size, d)

    expanded_p_s2 = s2.view(t2, 1, batch_size, d)
    expanded_p_s2 = expanded_p_s2.expand(t2, t1, batch_size, d)

    if not cat_only[0] and not cat_only[1]:
        matrix = torch.cat((expanded_p_s1, expanded_p_s2), dim=3)
    elif not cat_only[0] and cat_only[1]:
        matrix = torch.cat((expanded_p_s1, expanded_p_s2, expanded_p_s1 * expanded_p_s2), dim=3)
    else:
        matrix = torch.cat((expanded_p_s1,
                            expanded_p_s2,
                            torch.abs(expanded_p_s1 - expanded_p_s2),
                            expanded_p_s1 * expanded_p_s2), dim=3)

    # matrix = torch.cat((expanded_p_s1,
    #                     expanded_p_s2), dim=3)

    return matrix

def max_along_time(inputs, lengths):
    """
    :param inputs: [T * B * D] 
    :param lengths:  [B]
    :return: [B * D] max_along_time
    """
    ls = list(lengths)

    b_seq_max_list = []
    for i, l in enumerate(ls):
        seq_i = inputs[:l, i, :]
        seq_i_max, _ = seq_i.max(dim=0)
        seq_i_max = seq_i_max.squeeze()
        b_seq_max_list.append(seq_i_max)

    return torch.stack(b_seq_max_list)


def text_conv1d(inputs, l1, conv_filter: nn.Linear, k_size, dropout=None, list_in=False,
                gate_way=True):
    """
    :param inputs: [T * B * D] 
    :param l1:  [B]
    :param conv_filter:  [k * D_in, D_out * 2]
    :param k_size:  
    :param dropout: 
    :param padding: 
    :param list_in: 
    :return: 
    """
    k = k_size
    batch_size = l1.size(0)
    d_in = inputs.size(2) if not list_in else inputs[0].size(1)
    unit_d = conv_filter.out_features // 2
    pad_n = (k - 1) // 2

    zeros_padding = Variable(inputs[0].data.new(pad_n, d_in).zero_())

    batch_list = []
    input_list = []
    for b_i in range(batch_size):
        masked_in = inputs[:l1[b_i], b_i, :] if not list_in else inputs[b_i]
        if gate_way:
            input_list.append(masked_in)

        b_inputs = torch.cat([zeros_padding, masked_in, zeros_padding], dim=0)
        for i in range(l1[b_i]):
            # print(b_inputs[i:i+k])
            batch_list.append(b_inputs[i:i+k].view(k * d_in))

    batch_in = torch.stack(batch_list, dim=0)
    a, b = torch.chunk(conv_filter(batch_in), 2, 1)
    out = a * F.sigmoid(b)

    out_list = []
    start = 0
    for b_i in range(batch_size):
        if gate_way:
            out_list.append(torch.cat((input_list[b_i], out[start:start + l1[b_i]]), dim=1))
        else:
            out_list.append(out[start:start + l1[b_i]])

        start = start + l1[b_i]

    # max_out_list = []
    # for b_i in range(batch_size):
    #     max_out, _ = torch.max(out_list[b_i], dim=0)
    #     max_out_list.append(max_out)
    # max_out = torch.cat(max_out_list, 0)
    #
    # print(out_list)

    return out_list