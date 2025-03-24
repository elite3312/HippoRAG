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
  #export HF_HOME=<path to Huggingface home directory>
  export OPENAI_API_KEY=<open ai key>   # if you want to use OpenAI model
  conda activate hipporag
  python example.py
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```

- running on local
  ```txt
  (hipporag) (base) perry@DESKTOP-LGGEMNE:~/nlp2025/rag_aura_gang$ /home/perry/miniconda3/envs/hipporag/bin/python /home/perry/nlp2025/rag_aura_gang/example.py
  Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.21s/it]
  Some parameters are on the meta device because they were offloaded to the disk and cpu.
  9it [00:00, 241979.08it/s]
  9it [00:00, 618831.74it/s]
  /home/perry/.cache/huggingface/modules/transformers_modules/nvidia/NV-Embed-v2/c50d55f43bde7e6a18e0eaa15a62fd63a930f1a1/modeling_nvembed.py:349: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
    'input_ids': torch.tensor(batch_dict.get('input_ids').to(batch_dict.get('input_ids')).long()),
  /home/perry/miniconda3/envs/hipporag/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
    self.gen = func(*args, **kwds)
  Retrieving: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  4.79it/s]
  Collecting QA prompts: 100%|██████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 472.74it/s]
  QA Reading: 100%|████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 1310.86it/s]
  Extraction Answers from LLM Response: 3it [00:00, 40201.00it/s]
  Retrieving: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 639.51it/s]
  Collecting QA prompts: 100%|████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 45756.04it/s]
  QA Reading: 100%|████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 2631.31it/s]
  Extraction Answers from LLM Response: 3it [00:00, 78154.73it/s]
  Retrieving: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 677.05it/s]
  Length of retrieved docs (9) is smaller than largest topk for recall score (200)
  Length of retrieved docs (9) is smaller than largest topk for recall score (200)
  Length of retrieved docs (9) is smaller than largest topk for recall score (200)
  Collecting QA prompts: 100%|████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 36900.04it/s]
  QA Reading: 100%|████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 3014.59it/s]
  Extraction Answers from LLM Response: 3it [00:00, 202950.19it/s]
  ([QuerySolution(question="What is George Rankin's occupation?", docs=['George Rankin is a politician.', 'Thomas Marwick is a politician.', 'Oliver Badman is a politician.', 'Montebello is a part of Rockland County.', 'Marina is bom in Minsk.', "Erik Hort's birthplace is Montebello.", 'The prince used the lost glass slipper to search the kingdom.', 'When the slipper fit perfectly, Cinderella was reunited with the prince.', 'Cinderella attended the royal ball.'], doc_scores=array([1.        , 0.20697364, 0.19467027, 0.12020551, 0.07987539,
        0.0749496 , 0.07235771, 0.05784214, 0.        ], dtype=float32), answer='Politician.', gold_answers=['Politician'], gold_docs=['George Rankin is a politician.']), QuerySolution(question='How did Cinderella reach her happy ending?', docs=['When the slipper fit perfectly, Cinderella was reunited with the prince.', 'Cinderella attended the royal ball.', 'The prince used the lost glass slipper to search the kingdom.', 'Marina is bom in Minsk.', 'Montebello is a part of Rockland County.', 'Thomas Marwick is a politician.', 'George Rankin is a politician.', "Erik Hort's birthplace is Montebello.", 'Oliver Badman is a politician.'], doc_scores=array([1.47723310e-01, 8.07949975e-03, 4.42189436e-03, 1.02747368e-03,
        5.21765230e-04, 4.41726893e-04, 3.11764007e-04, 3.32584488e-05,
        0.00000000e+00]), answer='Cinderella reached her happy ending by having the glass slipper fit perfectly, reuniting her with the prince after attending the royal ball.', gold_answers=['By going to the ball.'], gold_docs=['Cinderella attended the royal ball.', 'The prince used the lost glass slipper to search the kingdom.', 'When the slipper fit perfectly, Cinderella was reunited with the prince.']), QuerySolution(question="What county is Erik Hort's birthplace a part of?", docs=['Montebello is a part of Rockland County.', "Erik Hort's birthplace is Montebello.", 'Marina is bom in Minsk.', 'George Rankin is a politician.', 'The prince used the lost glass slipper to search the kingdom.', 'Thomas Marwick is a politician.', 'Oliver Badman is a politician.', 'When the slipper fit perfectly, Cinderella was reunited with the prince.', 'Cinderella attended the royal ball.'], doc_scores=array([1.43240835e-01, 1.21890320e-02, 1.96095421e-03, 1.26009426e-03,
        1.14073008e-03, 9.55475455e-04, 2.40689053e-04, 1.36937947e-04,
        0.00000000e+00]), answer='Rockland County.', gold_answers=['Rockland County'], gold_docs=["Erik Hort's birthplace is Montebello.", 'Montebello is a part of Rockland County.'])], ['The text states that George Rankin is a politician. Therefore, his occupation is clearly defined as such. There are no additional details or context needed to determine his occupation from the provided information. \nAnswer: Politician.', "To determine how Cinderella reached her happy ending, I need to analyze the provided passages related to her story. The first passage indicates that Cinderella was reunited with the prince when the slipper fit perfectly, which suggests that the fitting of the slipper was a crucial moment leading to her happy ending. The second passage states that Cinderella attended the royal ball, which is likely where she first met the prince. The third passage mentions that the prince used the lost glass slipper to search the kingdom, indicating that the slipper was instrumental in finding Cinderella again. \n\nPutting these pieces together, Cinderella's happy ending was achieved through the combination of attending the royal ball, the prince searching for her using the glass slipper, and ultimately, the slipper fitting her perfectly, leading to their reunion.\n\nAnswer: Cinderella reached her happy ending by having the glass slipper fit perfectly, reuniting her with the prince after attending the royal ball.", "Erik Hort's birthplace is Montebello, which is stated to be a part of Rockland County. Therefore, Erik Hort's birthplace is also part of Rockland County. \nAnswer: Rockland County."], [{'prompt_tokens': 742, 'completion_tokens': 45, 'finish_reason': 'stop'}, {'prompt_tokens': 752, 'completion_tokens': 179, 'finish_reason': 'stop'}, {'prompt_tokens': 752, 'completion_tokens': 44, 'finish_reason': 'stop'}], {'Recall@1': 0.6111, 'Recall@2': 0.8889, 'Recall@5': 1.0, 'Recall@10': 1.0, 'Recall@20': 1.0, 'Recall@30': 1.0, 'Recall@50': 1.0, 'Recall@100': 1.0, 'Recall@150': 1.0, 'Recall@200': 1.0}, {'ExactMatch': 0.6667, 'F1': 0.7246})
  (hipporag) (base) perry@DESKTOP-LGGEMNE:~/nlp2025/rag_aura_gang$ 
  ```
- running one google compute engine
  
  ```txt
  
  (hipporag) perrywu12@instance-20250323-150245:~/HippoRAG$ python example.py 
  Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 4/4 [01:29<00:00, 22.46s/it]
  Batch Encoding:   0%|                                                                      | 0/9 [00:00<?, ?it/s]/home/perrywu12/.cache/huggingface/modules/transformers_modules/nvidia/NV-Embed-v2/c50d55f43bde7e6a18e0eaa15a62fd63a930f1a1/modeling_nvembed.py:349: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
    'input_ids': torch.tensor(batch_dict.get('input_ids').to(batch_dict.get('input_ids')).long()),
  /home/perrywu12/miniconda3/envs/hipporag/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
    self.gen = func(*args, **kwds)
  Batch Encoding: 10it [00:01,  8.00it/s]                                                                          
  NER: 100%|█| 9/9 [00:01<00:00,  4.73it/s, total_prompt_tokens=1440, total_completion_tokens=146, num_cache_hit=0]
  Extracting triples: 100%|█| 9/9 [00:01<00:00,  4.74it/s, total_prompt_tokens=4190, total_completion_tokens=270, n
  Batch Encoding: 20it [00:00, 58.66it/s]                                                                          
  Batch Encoding: 15it [00:00, 47.90it/s]                                                                          
  9it [00:00, 64972.01it/s]
  9it [00:00, 166293.99it/s]
  KNN for Queries: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 17.46it/s]
  100%|█████████████████████████████████████████████████████████████████████████| 16/16 [00:00<00:00, 68548.38it/s]
  {'num_phrase_nodes': 16, 'num_passage_nodes': 9, 'num_total_nodes': 25, 'num_extracted_triples': 13, 'num_triples_with_passage_node': 20, 'num_synonymy_triples': 13, 'num_total_triples': 46}
  Retrieving: 100%|██████████████████████████████████████████████████████████████████| 3/3 [00:04<00:00,  1.38s/it]
  Collecting QA prompts: 100%|████████████████████████████████████████████████████| 3/3 [00:00<00:00, 20493.34it/s]
  QA Reading: 100%|██████████████████████████████████████████████████████████████████| 3/3 [00:05<00:00,  1.97s/it]
  Extraction Answers from LLM Response: 3it [00:00, 93902.33it/s]
  Retrieving: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 367.08it/s]
  Collecting QA prompts: 100%|████████████████████████████████████████████████████| 3/3 [00:00<00:00, 24151.46it/s]
  QA Reading: 100%|██████████████████████████████████████████████████████████████████| 3/3 [00:05<00:00,  1.83s/it]
  Extraction Answers from LLM Response: 3it [00:00, 88612.06it/s]
  Retrieving: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 367.19it/s]
  Length of retrieved docs (9) is smaller than largest topk for recall score (200)
  Length of retrieved docs (9) is smaller than largest topk for recall score (200)
  Length of retrieved docs (9) is smaller than largest topk for recall score (200)
  Collecting QA prompts: 100%|████████████████████████████████████████████████████| 3/3 [00:00<00:00, 26269.13it/s]
  QA Reading: 100%|████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 2445.66it/s]
  Extraction Answers from LLM Response: 3it [00:00, 106634.85it/s]
  ([QuerySolution(question="What is George Rankin's occupation?", docs=['George Rankin is a politician.', 'Thomas Marwick is a politician.', 'Oliver Badman is a politician.', 'Montebello is a part of Rockland County.', 'Marina is bom in Minsk.', "Erik Hort's birthplace is Montebello.", 'The prince used the lost glass slipper to search the kingdom.', 'When the slipper fit perfectly, Cinderella was reunited with the prince.', 'Cinderella attended the royal ball.'], doc_scores=array([1.04457601e-01, 2.88495234e-02, 2.86234344e-02, 2.36941484e-03,
        1.59967606e-03, 1.53991970e-03, 1.35484696e-03, 1.10035870e-03,
        2.94879917e-05]), answer='Politician.', gold_answers=['Politician'], gold_docs=['George Rankin is a politician.']), QuerySolution(question='How did Cinderella reach her happy ending?', docs=['When the slipper fit perfectly, Cinderella was reunited with the prince.', 'Cinderella attended the royal ball.', 'The prince used the lost glass slipper to search the kingdom.', 'Marina is bom in Minsk.', 'Montebello is a part of Rockland County.', 'Thomas Marwick is a politician.', 'George Rankin is a politician.', "Erik Hort's birthplace is Montebello.", 'Oliver Badman is a politician.'], doc_scores=array([6.21493069e-02, 4.42779252e-02, 1.59776678e-02, 9.40619669e-04,
        4.01773756e-04, 3.88396911e-04, 2.79620656e-04, 4.68148938e-05,
        1.99408229e-05]), answer='Cinderella reached her happy ending by having the glass slipper fit perfectly, reuniting her with the prince after attending the royal ball.', gold_answers=['By going to the ball.'], gold_docs=['Cinderella attended the royal ball.', 'The prince used the lost glass slipper to search the kingdom.', 'When the slipper fit perfectly, Cinderella was reunited with the prince.']), QuerySolution(question="What county is Erik Hort's birthplace a part of?", docs=["Erik Hort's birthplace is Montebello.", 'Montebello is a part of Rockland County.', 'Marina is bom in Minsk.', 'George Rankin is a politician.', 'The prince used the lost glass slipper to search the kingdom.', 'Thomas Marwick is a politician.', 'Oliver Badman is a politician.', 'When the slipper fit perfectly, Cinderella was reunited with the prince.', 'Cinderella attended the royal ball.'], doc_scores=array([9.73987213e-02, 5.92261146e-02, 2.62042617e-03, 1.63159201e-03,
        1.41936170e-03, 1.25971537e-03, 3.88207685e-04, 1.54272202e-04,
        4.93964848e-06]), answer='Rockland County.', gold_answers=['Rockland County'], gold_docs=["Erik Hort's birthplace is Montebello.", 'Montebello is a part of Rockland County.'])], ['The text states that George Rankin is a politician. Therefore, his occupation is clearly defined as such. There are no additional details or context needed to determine his occupation from the provided information. \nAnswer: Politician.', "To determine how Cinderella reached her happy ending, I need to analyze the provided passages related to her story. The first passage indicates that Cinderella was reunited with the prince when the slipper fit perfectly, which suggests that the fitting of the slipper was a crucial moment leading to her happy ending. The second passage states that Cinderella attended the royal ball, which is likely where she first met the prince. The third passage mentions that the prince used the lost glass slipper to search the kingdom, indicating that the slipper was instrumental in finding Cinderella again. \n\nCombining these elements, Cinderella's happy ending was achieved through the sequence of attending the royal ball, the prince searching for her using the lost glass slipper, and ultimately, the slipper fitting her perfectly, leading to their reunion.\n\nAnswer: Cinderella reached her happy ending by having the glass slipper fit perfectly, reuniting her with the prince after attending the royal ball.", "Erik Hort's birthplace is Montebello. According to the information provided, Montebello is a part of Rockland County. Therefore, Erik Hort's birthplace, Montebello, is also part of Rockland County. \nAnswer: Rockland County."], [{'prompt_tokens': 742, 'completion_tokens': 45, 'finish_reason': 'stop'}, {'prompt_tokens': 752, 'completion_tokens': 180, 'finish_reason': 'stop'}, {'prompt_tokens': 752, 'completion_tokens': 54, 'finish_reason': 'stop'}], {'Recall@1': 0.6111, 'Recall@2': 0.8889, 'Recall@5': 1.0, 'Recall@10': 1.0, 'Recall@20': 1.0, 'Recall@30': 1.0, 'Recall@50': 1.0, 'Recall@100': 1.0, 'Recall@150': 1.0, 'Recall@200': 1.0}, {'ExactMatch': 0.6667, 'F1': 0.7246})
  ```
## notes

- HippoRAG/main.py has some configs that we can tune, as well as changing the dataset
- basic config contains more hyperparameters
- running on local is quite expensive
- possible topics
  - create our own dataset
  - change hyper parameters
    - src/hipporag/utils/config_utils.py
- gcp image
  - c0-deeplearning-common-cu124-v20250310-debian-11-py310
- nvembedv2.py "torch_dtype": "auto", in _init_embedding_config() can reduce the size of the embedding model on the gpu
