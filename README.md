# hipporag

## install

```sh
#conda bin is at:
#/home/perry/miniconda3/bin
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
conda create -n hipporag python=3.10 # create a venv called hipporag
conda activate hipporag

git clone https://github.com/elite3312/HippoRAG.git
cd HippoRAG
pip install hipporag # this somehow installs pytorch==2.5.1, if cuda is too old need to downgrade
#conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia # cuda 12.4
#conda env remove --name lightrag
watch nvidia-smi
```

## running experiments

- set environment variables
  
  ```sh
  export CUDA_VISIBLE_DEVICES=0,1,2,3 # change according to how many gpus you have
  #export HF_HOME=<path to Huggingface home directory> # this can be omitted, and the data will simply be in .cache
  export OPENAI_API_KEY=<open ai key>   # if you want to use OpenAI model
  conda activate hipporag
  python example.py
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python main.py --dataset itsupport
  ```
### exp1-tuning config

```sh
python main.py --dataset itsupport
```

### exp2-change embedding models

```sh
python main.py --dataset itsupport --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2
python main.py --dataset itsupport --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name GritLM/GritLM-7B
python main.py --dataset itsupport --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name facebook/contriever 
```

### exp3-change prompt

```sh
python main.py --dataset itsupport --ner_setting ner_it_multi
```
