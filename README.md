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
After first experiment on RoBERTa for bayesian optimization, we found that the model works best with learning rate between (1e-5, 3e-5) and we set search space={learning_rate:(1e-5, 3e-5)} or ={batch_size:[8, 32], learning_rate:(1e-5, 3e-5)} respectively. The graph also shows that larger learning rates work better with the model, while 5e05 might be too small for model to learn.

![DS301_project](/ro_bayes.png)

The overall performance of our 9 experiments are shown below. Here the index 1 indicates Grid search, index 2 indicates Random search, and 3 indicates Bayesian search. Our experiment shows that after finetuning RoBERTa improves performance up to 0.81, and DistilBERT outperforms BERT and has similar perforamnce to RoBERTa dispite that DistilBERT has only one third of the size of the latter two. (BERT with grid search has evaluation accuracy=0 because grid search failed to find BERT an optimal configuration under the given number of trials and hyperparameter setting.)

![DS301_project](/performance.png)

We also compared training time for each trial of different models. Here the index indicates the same as in the previous chart. DistilBERT again beats the large models by requiring only about half of the training time per trial.

![DS301_project](/time.png)
