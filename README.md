# gpt2-fine-tuning
*Assignment 3* of *Advanced Natural Language Processing* in IIIT-Hyderabad, Monsoon '24. 

For more details, see the [report](./docs/Report.pdf). 

## setting up the dependencies
the `conda` env files are available in [the docs directory](./docs/), with the output of `conda env export` in [envs](./docs/envs.yml) and the result of `conda env export --from-history` in [envs-hist](./docs/envs-hist.yml). 

You may set up your environment by using:
```sh
conda env create -f docs/envs-hist.yml
```

Or, include the following packages in your environment:
- pytorch
- pytorch-cuda=12.4 (change according to your CUDA version)
- torchaudio (probably redundant)
- torchvision (probably redundant)
- evaluate
- kagglehub
- transformers[version='>=3.5']
- datasets==2.10.0
- rouge-score
- peft

## downloading the model, tokenizer and data

### the data
The data is the [cnn-dailymail dataset on text summarisation](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail). It can be downloaded locally into the [./data](./data/) directory using:

```sh
python scripts/download-data.py
```

### the model and tokenizer 
The model and corresponding tokenizer of choice is that of [gpt-neo](https://huggingface.co/EleutherAI/gpt-neo-125m). They can be downloaded locally to [./model](./model/) by running:

```sh
python scripts/download-model.py
```

## fine tuning
To choose the type of fine-tuning to train and test on, alter the enumerator passed to the corresponding function in the `main` call of [train](./src/train.py) and [test](./src/test.py) respectively. The hyper-parameters, too, can be altered in the arguments to this call. 

The reason for choosing this rather than command line arguments was to allow easy copying of the scripts over to a Kaggle notebook. 

### training
To fine-tune the model, run:
```sh
python -m src.train
```
the fine-tuned checkpoint will be saved to `traditional.pth`, `lora.pth` or `soft_prompts.pth` depending on the fine-tuning method of choice. 

### testing
To test the fine-tuned model, pass the path to the checkpoint and the fine tuning type enum in the call to the main function in [test](./src/test.py), then run:
```sh
python -m src.test
```

## fine-tuned models
The loss-per-epoch and loss details can be found in the [res directory](./res/) while the tuned checkpoints themselves can be found in [this drive link](https://drive.google.com/drive/folders/1V3wf21I0HAq0Zu1b4B04rqkJKDBWX0y7?usp=sharing). 

To test these models, we can follow the same instructions above for testing. 

## todo
- [ ] mention all the fancy stuff done for mem efficiency and all 
- [ ] clean the data
  - check this: https://github.com/abisee/cnn-dailymail
- [ ] make a parent class that does the model loading and freezing in the `__init__`, and instructs to return the logits from the `forward` call (taking input ids and attention mask).
- [ ] possible optimisation mentioned in the test of [utils](./src/utils.py)
- [ ] update with new assignment doc 
