import os
# from util.data_loader import RParsedTextLField
# from util.data_loader import ParsedTextLField

from torchtext import data, vocab
from torchtext import datasets

import config
import torch


class MNLI(data.ZipDataset, data.TabularDataset):
    # url = 'http://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    filename = 'multinli_0.9.zip'
    dirname = 'multinli_0.9'

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(
            len(ex.premise), len(ex.hypothesis))

    @classmethod
    def splits(cls, text_field, label_field, parse_field=None, genre_field=None, root='.',
               train=None, validation=None, test=None):
        """Create dataset objects for splits of the SNLI dataset.
        This is the most flexible way to use the dataset.
        Arguments:
            text_field: The field that will be used for premise and hypothesis
                data.
            label_field: The field that will be used for label data.
            parse_field: The field that will be used for shift-reduce parser
                transitions, or None to not include them.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose snli_1.0
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.jsonl'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'dev.jsonl'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'test.jsonl'.
        """
        path = cls.download_or_unzip(root)
        if parse_field is None:
            return super(MNLI, cls).splits(
                os.path.join(path, 'multinli_0.9_'), train, validation, test,
                format='json', fields={'sentence1': ('premise', text_field),
                                       'sentence2': ('hypothesis', text_field),
                                       'gold_label': ('label', label_field)},
                filter_pred=lambda ex: ex.label != '-')
        return super(MNLI, cls).splits(
            os.path.join(path, 'multinli_0.9_'), train, validation, test,
            format='json', fields={'sentence1_binary_parse':
                                   [('premise', text_field),
                                    ('premise_transitions', parse_field)],
                                   'sentence2_binary_parse':
                                   [('hypothesis', text_field),
                                    ('hypothesis_transitions', parse_field)],
                                   'gold_label': ('label', label_field),
                                   'genre': ('genre', genre_field)},
            filter_pred=lambda ex: ex.label != '-')

if __name__ == "__main__":
    # pass
    #
    testr_field = ParsedTextLField()
    transitions_field = datasets.snli.ShiftReduceField()
    y_field = data.Field(sequential=False)
    g_field = data.Field(sequential=False)

    data_root = config.DATA_ROOT

    snli_train, snli_dev, snli_test = datasets.SNLI.splits(testr_field, y_field, transitions_field, root=data_root)

    mnli_train, mnli_dev_m, mnli_dev_um = MNLI.splits(testr_field, y_field, transitions_field, g_field, root=data_root,
                                                      train='train.jsonl',
                                                      validation='dev_matched.jsonl',
                                                      test='dev_mismatched.jsonl')

    mnli_test_m, mnli_test_um = MNLI.splits(testr_field, y_field, transitions_field, g_field, root=data_root,
                                            train=None,
                                            validation='test_matched_unlabeled.jsonl',
                                            test='test_mismatched_unlabeled.jsonl')

    testr_field.build_vocab(snli_train, snli_dev, snli_test,
                            mnli_train, mnli_dev_m, mnli_dev_um, mnli_test_m, mnli_test_um)

    # print(testr_field.vocab.freqs)

    g_field.build_vocab(mnli_train, mnli_dev_m, mnli_dev_um, mnli_test_m, mnli_test_um)
    y_field.build_vocab(snli_train)
    #
    snli_train_iter, snli_dev_iter, snli_test_iter, mnli_train_iter, mnli_dev_m_iter, mnli_dev_um_iter, mnli_test_m_iter, mnli_test_um_iter \
        = data.Iterator.splits(
        (snli_train, snli_dev, snli_test, mnli_train, mnli_dev_m, mnli_dev_um, mnli_test_m, mnli_test_um), batch_sizes=(3, 3, 3, 3, 3, 3, 3, 3),
         device=-1, shuffle=False, sort=False)

    # print(y_field.vocab.freqs)
    # print(g_field.vocab.freqs)
    #
    # print(g_field.vocab.stoi)
    # print(g_field.vocab.stoi)
    #
    # print(len(train))
    # print(len(dev_m))
    # print(vars(mnli_train[0]))

    print(vars(mnli_dev_um[0]))
    print(vars(mnli_dev_um[1]))
    print(vars(mnli_dev_um[2]))

    batch = next(iter(mnli_dev_um_iter))

    s2, s2_l = batch.hypothesis
    # s2_t = batch.hypothesis_transitions
    y = batch.label

    print('s2', s2, s2_l - 1)
    # print('s2_t', s2_t)
    print('label', y)

    # testr_field.vocab.load_vectors(wv_dir='/Users/Eason/RA/SSU/tst/', wv_type='glove.840B', wv_dim=300)
    # torch.save(testr_field.vocab.vectors, './saved_embd.pt')
    # #
    # print(testr_field.vocab.stoi)
    # print(testr_field.vocab.itos)
    # print(len(testr_field.vocab.freqs))
    # print(testr_field.vocab.freqs['<unk>'])

