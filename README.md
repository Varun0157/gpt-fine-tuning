# gpt2-fine-tuning
Advanced NLP (IIIT-H Monsoon '24) Assignment 3

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

## Downloading the model, tokenizer and data

### the data


