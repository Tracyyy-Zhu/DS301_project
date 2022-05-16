# DS301 Project
This is the course final project for DS-UA 301 Advanced Topics in Data Science: Practical ML/DL tools and techniques. 

# Team Members
[Yukai Yang](https://github.com/yk803) \
[Tracy Zhu](https://github.com/Tracyyy-Zhu) \
To download the whole directory, enter 

    git clone https://github.com/Tracyyy-Zhu/DS301_project.git 

in terminal.

# Description
This project explores the effect of fine-tuning large language models on their performances of downstream tasks. For a question answering task using [BoolQ](https://github.com/google-research-datasets/boolean-questions.git) dataset, we fine-tuned RoBERTa, BERT, and DistilBERT with three hyperparameter optimization techniques: Grid search, Random search, and Bayesian optimization search.

# Code structure
utils directory contains files for encoding the data using `torch.utils.data.Dataset` and `transformers.PreTrainedTokenizerFast`.

experiments directory includes three directories for RoBERTa, BERT, and DistilBERT each, containing a `model_finetune.py` for constructing the customized metrics and initialized pre-trained model. The remaining files in the directory are as their names suggested, experiment scripts for the model on respective hyperparmeter optimization algorithms each with a slurm file to run on hpc.

# Results and Observations

