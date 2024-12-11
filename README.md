# gpt2-fine-tuning
*Assignment 3* of *Advanced Natural Language Processing* in IIIT-Hyderabad, Monsoon '24. 

Implementing three fine-tuning methods on the summarisation task using the gpt-neo (125M) model, in particular - prompt tuning usnig soft prompts, LoRA, and traditional fine tuning using only the last classifier layer of the model.

In addition, gradient accumulation has been implemented in order to effectively increase the batch size without requiring additional memory. 

For more details including performance comparisons, see the [report](./docs/Report.pdf). 

## details 

### prompt tuning 
Prepend soft prompt tokens to our input sequences and backpropogate only throughthe soft prompt embeddings,keeping the gpt-neo parameters frozen. We initialise with tokens of the form "SUMMARIZE". 

### LoRA Fine Tuning 
Implement LoRA by adding low-rank matrices wherever possible to adapt th gpt-neo parameters efficiently. We use in-built libraries for this. 

### Traditional Fine Tuning (last layers only)
Fine tune only the last few layers of the gpt-neo model. 

## setting up the dependencies
the `conda` env files are available in [the docs directory](./docs/), with the output of `conda env export` in [envs](./docs/envs.yml) and the result of `conda env export --from-history` in [envs-hist](./docs/envs-hist.yml). 

You may set up your environment by using:
```sh
conda env create -f docs/envs.yml
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

The dependencies exported from history [are also available](./docs/envs-hist.yml). 

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
Choose the method of fine tuning using the `--fine_tuning_type` cli argument. Hyperparameters can be modified directly in the code. 

### training
To fine-tune the model, run:
```sh
python -m src.train --fine_tuning_type <type>
```
the fine-tuned checkpoint will be saved to `traditional.pth`, `lora.pth` or `soft_prompts.pth` depending on the fine-tuning method of choice. 

### testing
To test the fine-tuned model, pass the fine tuning type as a command line argument:
```sh
python -m src.test --fine_tuning_type <type>
```

## fine-tuned models
The loss-per-epoch and loss details can be found in the [res directory](./res/) while the tuned checkpoints themselves can be found in [this drive link](https://drive.google.com/drive/folders/1V3wf21I0HAq0Zu1b4B04rqkJKDBWX0y7?usp=sharing). 

To test these models, we can follow the same instructions above for testing. 

## todo
- [ ] possible optimisation mentioned in the test of [utils](./src/utils.py)
- [ ] re-run models and save in drive - change in model structure could lead to issues while loading 
