# multiNLI_encoder
This is a repo for multiNLI_encoder.

**Note:**
This repo is about Shortcut-Stacked Sentence Encoders for the MultiNLI dataset. We recommend users to check this new repo: (https://github.com/easonnie/ResEncoder)[https://github.com/easonnie/ResEncoder], especially if you are interested in SNLI or Residual-Stacked Sentence Encoders. This new encoder achieves almost same results as the shortcut-stacked one with much fewer parameters. See (https://arxiv.org/abs/1708.02312)[https://arxiv.org/abs/1708.02312] (Section 6) for comparing results.

Try to follow the instruction below to successfully run the experiment.

1.Download the additional `data.zip` file, unzip it and place it at the root directory of this repo.
Link for download `data.zip` file: [*DropBox Link*](https://www.dropbox.com/sh/kq81vmcmwktlyji/AADRVQRh9MdcXTkTQct7QlQFa?dl=0)

2.This repo is based on an old version of `torchtext`, the latest version of `torchtext` is not backward-compatible.
We provide a link to download the old `torchtext` that should be used for this repo. Link: [*old_torchtext*](https://www.dropbox.com/sh/n8ipkm1ng8f6d5u/AADg4KhwQMwz4xFkVJafgUMma?dl=0)

3.Install the required package below:
```
torchtext # The one you just download. Or you can use the latest torchtext by fixing the SNLI path problem.
pytorch
fire
tqdm
numpy
```

4.At the root directory of this repo, create a directory called `saved_model` by running the script below:
```
mkdir saved_model
```
This directory will be used for saving the models that produce best dev result.
Before running the experiments, make sure that the structure of this repo should be something like below.
```
├── config.py
├── data
│   ├── info.txt
│   ├── multinli_0.9
│   │   ├── multinli_0.9_dev_matched.jsonl
│   │   ├── multinli_0.9_dev_mismatched.jsonl
│   │   ├── multinli_0.9_test_matched_unlabeled.jsonl
│   │   ├── multinli_0.9_test_mismatched_unlabeled.jsonl
│   │   └── multinli_0.9_train.jsonl
│   ├── saved_embd.pt
│   └── snli_1.0
│       ├── snli_1.0_dev.jsonl
│       ├── snli_1.0_test.jsonl
│       └── snli_1.0_train.jsonl
├── model
│   ├── baseModel.py
│   └── tested_model
│       └── stack_3bilstm_last_encoder.py
├── README.md
├── saved_model
├── setup.sh
├── torch_util.py
└── util
    ├── data_loader.py
    ├── __init__.py
    ├── mnli.py
    └── save_tool.py
```

5.Start training by run the script in the root directory.
```
source setup.sh
python model/tested_model/stack_3bilstm_last_encoder.py
```

6.After training completed, there will be a folder created by the script in the `saved_model` directory that you created in step 3.
The parameters of the model will be saved in that folder. The path of the model will be something like:
```
$DIR_TMP/saved_model/(TIME_STAMP)_[512,1024,2048]-3stack-bilstm-last_maxout/saved_params/(YOUR_MODEL_WITH_DEV_RESULT)
```
Remember to change the bracketed part to the actual file name on your computer.

7.Now, you can evaluate the model on dev set again by running the script below.
```
python model/tested_model/stack_3bilstm_last_encoder.py eval_model "$DIR_TMP/saved_model/(TIME_STAMP)_[512,1024,2048]-3stack-bilstm-last_maxout/saved_params/(YOUR_MODEL_WITH_DEV_RESULT)"
```
