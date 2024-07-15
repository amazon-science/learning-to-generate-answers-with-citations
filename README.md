## Description

This repository maintains the code for **CaLF**, with the associated ACL 2024 paper: [Learning to Generate Answers with Citations via Factual Consistency Models](https://arxiv.org/abs/2406.13124).

> Large Language Models (LLMs) frequently hallucinate, impeding their reliability in mission-critical situations. One approach to address this issue is to provide citations to relevant sources alongside generated content, enhancing the verifiability of generations. However, citing passages accurately in answers remains a substantial challenge. This paper proposes a weakly-supervised fine-tuning method leveraging factual consistency models (FCMs). Our approach alternates between generating texts with citations and supervised fine-tuning with FCM-filtered citation data. Focused learning is integrated into the objective, directing the fine-tuning process to emphasise the factual unit tokens, as measured by an FCM. Results on the ALCE few-shot citation benchmark with various instruction-tuned LLMs demonstrate superior performance compared to in-context learning, vanilla supervised fine-tuning, and state-of-the-art methods,  with an average improvement of $34.1$, $15.5$, and $10.5$ citation F$_1$ points, respectively. Moreover, in a domain transfer setting we show that the obtained citation generation ability robustly transfers to unseen datasets. Notably, our citation improvements contribute to the lowest factual error rate across baselines.

## Installation

Install relevant packages within a new conda environment by calling the installation script from the root directory:
```
./bin/installation/install.sh
```

Download the AlignScore factual consistency model and place it into the expected folder by calling the following script:

```
./bin/installation/download_fcm.sh
```


### Download Data

Sample data from the ASQA dataset to run CaLF out of the box is part of the repository in `data/asqa/`. CaLF can be trained on any long-form question answering dataset. To train a model on the full data for `ASQA`, `ELI5`, and `FactScore` as used for the experiments described in the paper, please download the data [here](https://drive.google.com/file/d/1VulWcG80vQ6V7TZcq4kflitE5Xb7IHvE/view?usp=sharing).


## Running Experiments

To check first whether the code pipeline will execute correcty, you can run CaLF on a small subset of the data by calling

```
./bin/run_calf_debug.sh default asqa lora_100_steps_bootstrapping_chat_templates token_rescaling mistralorca gtr all alignscore_threshold_09 0 0,1,2,3,4,5,6,7
```

To train and evaluate CaLF on the full ASQA dataset as provided in `data/asqa` run:

```
./bin/run_calf.sh default asqa lora_100_steps_bootstrapping_chat_templates token_rescaling mistralorca gtr all alignscore_threshold_09 0 0,1,2,3,4,5,6,7
```

1. `default`: answer truncation mode (none)
2. `asqa`: dataset
3. `lora_100_steps_bootstrapping_chat_templates`: environment (lora fine-tuning, 100 steps, chat templates, and in-context in first iteration)
4. `token_rescaling`: focused learning mode (either `token_rescaling` or `none`)
6. `mistralorca`: LLM
7. `gtr`: Retrieval system (the retrieval is pre-compiled with ASQA only supporting GTR, and ELI5 only supporting BM25)
8. `all`: Training samples to use
9. `alignscore_threshold_09`: Generation of weakly-supervised data with filtering via AlignScore and a threshold of 0.9
10. `0`: Random seed
11. `0,1,2,3,4,5,6,7`: CUDA visible devices

All configuration settings for each argument can be found in the folder `configs`.

The script trains the LLM iteratively on a fully and weakly supervsed data and evaluates its performance after the training process is completed. 

If you want to call the evaluation script independently, call `val.sh` with the same arguments as above. If you have trained a CaLF model which you wish to evaluate, you can call `val_from_saved.sh`, which loads the trained weights before evaluation. Finally, `val_from_saved_transfer.sh` can be used to evaluate a model on a new dataset in a domain-transfer scenario (i.e. results in Table 2). In addition to the afforementioned arguments, the transfer evaluation script takes two additional arguments: the target dataset name and the target dataset retrieval system (13 arguments in total).

### Results
CaLF stops training on the sample data after 7 iterations, producing the following results:


| Model                  | Rougle-L | EM Recall (Grounded) | Citation F1 |
|------------------------|----------|----------------------|-------------|
| Baseline (sample data) | 38.2     | 29.0                 | 72.6        |
| CaLF (sample data)     | 40.3     | 28.9                 | 81.4        |
| CaLF                   | 40.9     | 34.5                 | 80.5        |

Already with the 240 unlabeled training instances of the sample data (1/4 of the full training data), we observe substential citation improvements compared to our FT-baseline. However, for best results train on the entire data collection (see above).


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
