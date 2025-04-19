# experimental results

## passage_node_weight

- 0.5
  
  ```txt
  (hipporag) perrywu12@instance-20250323-150245:~/HippoRAG$ python main.py --dataset itsupport --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2
    INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/perrywu12/HippoRAG/src/hipporag/prompts/templates
    INFO:datasets:PyTorch version 2.5.1 available.
    INFO:datasets:Polars version 1.26.0 available.
    INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
    Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 4/4 [01:29<00:00, 22.48s/it]
    INFO:src.hipporag.embedding_store:Loaded 100 records from outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet
    INFO:src.hipporag.embedding_store:Loaded 1298 records from outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/entity_embeddings/vdb_entity.parquet
    INFO:src.hipporag.embedding_store:Loaded 1456 records from outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/fact_embeddings/vdb_fact.parquet
    INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/perrywu12/HippoRAG/src/hipporag/prompts/templates
    INFO:src.hipporag.HippoRAG:Indexing Documents
    INFO:src.hipporag.HippoRAG:Performing OpenIE
    INFO:src.hipporag.embedding_store:Inserting 0 new records, 100 records already exist.
    INFO:src.hipporag.HippoRAG:OpenIE results saved to outputs/itsupport/openie_results_ner_gpt-4o-mini.json
    INFO:src.hipporag.HippoRAG:Encoding Entities
    INFO:src.hipporag.embedding_store:Inserting 0 new records, 1298 records already exist.
    INFO:src.hipporag.HippoRAG:Encoding Facts
    INFO:src.hipporag.embedding_store:Inserting 0 new records, 1456 records already exist.
    INFO:src.hipporag.HippoRAG:Constructing Graph
    INFO:src.hipporag.HippoRAG:Adding OpenIE triples to graph.
    100it [00:00, 17179.21it/s]
    INFO:src.hipporag.HippoRAG:Connecting passage nodes to phrase nodes.
    100it [00:00, 45620.01it/s]
    INFO:src.hipporag.HippoRAG:Found 100 new chunks to save into graph.
    INFO:src.hipporag.HippoRAG:Expanding graph with synonymy edges
    INFO:src.hipporag.HippoRAG:Performing KNN retrieval for each phrase nodes (1298).
    KNN for Queries: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  2.93it/s]
    100%|█████████████████████████████████████████████████████████████████████| 1298/1298 [00:00<00:00, 89575.28it/s]
    INFO:src.hipporag.HippoRAG:Graph construction completed!
    {'num_phrase_nodes': 1298, 'num_passage_nodes': 100, 'num_total_nodes': 1398, 'num_extracted_triples': 1456, 'num_triples_with_passage_node': 1832, 'num_synonymy_triples': 5760, 'num_total_triples': 9048}
    INFO:src.hipporag.HippoRAG:Writing graph with 1398 nodes, 9045 edges
    INFO:src.hipporag.HippoRAG:Saving graph completed!
    INFO:src.hipporag.HippoRAG:Preparing for fast retrieval.
    INFO:src.hipporag.HippoRAG:Loading keys.
    INFO:src.hipporag.HippoRAG:Loading embeddings.
    INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_fact.
    Batch Encoding:   0%|                                                                     | 0/17 [00:00<?, ?it/s]/home/perrywu12/.cache/huggingface/modules/transformers_modules/nvidia/NV-Embed-v2/c50d55f43bde7e6a18e0eaa15a62fd63a930f1a1/modeling_nvembed.py:349: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
    'input_ids': torch.tensor(batch_dict.get('input_ids').to(batch_dict.get('input_ids')).long()),
    /home/perrywu12/miniconda3/envs/hipporag/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
    self.gen = func(*args, **kwds)
    Batch Encoding: 24it [00:01, 20.68it/s]                                                                          
    INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_passage.
    Batch Encoding: 24it [00:00, 48.96it/s]                                                                          
    Retrieving: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:02<00:00,  7.04it/s]
    INFO:src.hipporag.HippoRAG:Total Retrieval Time 4.17s
    INFO:src.hipporag.HippoRAG:Total Recognition Memory Time 0.29s
    INFO:src.hipporag.HippoRAG:Total PPR Time 2.04s
    INFO:src.hipporag.HippoRAG:Total Misc Time 1.84s
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    INFO:src.hipporag.HippoRAG:Evaluation results for retrieval: {'Recall@1': 0.0, 'Recall@2': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0, 'Recall@20': 0.0, 'Recall@30': 0.0, 'Recall@50': 0.0, 'Recall@100': 0.0, 'Recall@150': 0.0, 'Recall@200': 0.0}
    Collecting QA prompts: 100%|███████████████████████████████████████████████████| 17/17 [00:00<00:00, 6298.87it/s]
    QA Reading:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:01<00:00,  9.46it/s]
    Extraction Answers from LLM Response: 17it [00:00, 349525.33it/s]
    INFO:src.hipporag.HippoRAG:Evaluation results for QA: {'ExactMatch': 0.3529, 'F1': 0.7025}
    (hipporag) perrywu12@instance-20250323-150245:~/HippoRAG$ ^C
    (hipporag) perrywu12@instance-20250323-150245:~/HippoRAG$ git pull
    hint: Pulling without specifying how to reconcile divergent branches is
    hint: discouraged. You can squelch this message by running one of the following
    hint: commands sometime before your next pull:
    hint: 
    hint:   git config pull.rebase false  # merge (the default strategy)
    hint:   git config pull.rebase true   # rebase
    hint:   git config pull.ff only       # fast-forward only
    hint: 
    hint: You can replace "git config" with "git config --global" to set a default
    hint: preference for all repositories. You can also pass --rebase, --no-rebase,
    hint: or --ff-only on the command line to override the configured default per
    hint: invocation.
    remote: Enumerating objects: 10, done.
    remote: Counting objects: 100% (10/10), done.
    remote: Compressing objects: 100% (5/5), done.
    remote: Total 8 (delta 5), reused 6 (delta 3), pack-reused 0 (from 0)
    Unpacking objects: 100% (8/8), 2.98 KiB | 610.00 KiB/s, done.
    From https://github.com/elite3312/HippoRAG
    451437f..2eb8a49  test_perry -> origin/test_perry
    Already up to date.
    (hipporag) perrywu12@instance-20250323-150245:~/HippoRAG$ git checkout test_perry
    Branch 'test_perry' set up to track remote branch 'test_perry' from 'origin'.
    Switched to a new branch 'test_perry'
    (hipporag) perrywu12@instance-20250323-150245:~/HippoRAG$ git pull
    hint: Pulling without specifying how to reconcile divergent branches is
    hint: discouraged. You can squelch this message by running one of the following
    hint: commands sometime before your next pull:
    hint: 
    hint:   git config pull.rebase false  # merge (the default strategy)
    hint:   git config pull.rebase true   # rebase
    hint:   git config pull.ff only       # fast-forward only
    hint: 
    hint: You can replace "git config" with "git config --global" to set a default
    hint: preference for all repositories. You can also pass --rebase, --no-rebase,
    hint: or --ff-only on the command line to override the configured default per
    hint: invocation.
    Already up to date.
    (hipporag) perrywu12@instance-20250323-150245:~/HippoRAG$ python main.py --dataset itsupport --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2
    INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/perrywu12/HippoRAG/src/hipporag/prompts/templates
    INFO:src.hipporag.HippoRAG:Loaded graph from outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/graph.pickle with 1398 nodes, 9045 edges
    INFO:datasets:PyTorch version 2.5.1 available.
    INFO:datasets:Polars version 1.26.0 available.
    INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
    Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 4/4 [00:05<00:00,  1.45s/it]
    INFO:src.hipporag.embedding_store:Loaded 100 records from outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet
    INFO:src.hipporag.embedding_store:Loaded 1298 records from outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/entity_embeddings/vdb_entity.parquet
    INFO:src.hipporag.embedding_store:Loaded 1456 records from outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/fact_embeddings/vdb_fact.parquet
    INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/perrywu12/HippoRAG/src/hipporag/prompts/templates
    INFO:src.hipporag.HippoRAG:Indexing Documents
    INFO:src.hipporag.HippoRAG:Performing OpenIE
    INFO:src.hipporag.embedding_store:Inserting 0 new records, 100 records already exist.
    INFO:src.hipporag.HippoRAG:OpenIE results saved to outputs/itsupport/openie_results_ner_gpt-4o-mini.json
    INFO:src.hipporag.HippoRAG:Encoding Entities
    INFO:src.hipporag.embedding_store:Inserting 0 new records, 1298 records already exist.
    INFO:src.hipporag.HippoRAG:Encoding Facts
    INFO:src.hipporag.embedding_store:Inserting 0 new records, 1456 records already exist.
    INFO:src.hipporag.HippoRAG:Constructing Graph
    INFO:src.hipporag.HippoRAG:Adding OpenIE triples to graph.
    100it [00:00, 17328.97it/s]
    INFO:src.hipporag.HippoRAG:Connecting passage nodes to phrase nodes.
    100it [00:00, 1823610.43it/s]
    INFO:src.hipporag.HippoRAG:Preparing for fast retrieval.
    INFO:src.hipporag.HippoRAG:Loading keys.
    INFO:src.hipporag.HippoRAG:Loading embeddings.
    INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_fact.
    Batch Encoding:   0%|                                                                     | 0/17 [00:00<?, ?it/s]/home/perrywu12/.cache/huggingface/modules/transformers_modules/nvidia/NV-Embed-v2/c50d55f43bde7e6a18e0eaa15a62fd63a930f1a1/modeling_nvembed.py:349: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
    'input_ids': torch.tensor(batch_dict.get('input_ids').to(batch_dict.get('input_ids')).long()),
    /home/perrywu12/miniconda3/envs/hipporag/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
    self.gen = func(*args, **kwds)
    Batch Encoding: 24it [00:00, 24.82it/s]                                                                          
    INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_passage.
    Batch Encoding: 24it [00:00, 49.66it/s]                                                                          
    Retrieving: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:02<00:00,  7.54it/s]
    INFO:src.hipporag.HippoRAG:Total Retrieval Time 3.82s
    INFO:src.hipporag.HippoRAG:Total Recognition Memory Time 0.25s
    INFO:src.hipporag.HippoRAG:Total PPR Time 1.93s
    INFO:src.hipporag.HippoRAG:Total Misc Time 1.64s
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    INFO:src.hipporag.HippoRAG:Evaluation results for retrieval: {'Recall@1': 0.0, 'Recall@2': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0, 'Recall@20': 0.0, 'Recall@30': 0.0, 'Recall@50': 0.0, 'Recall@100': 0.0, 'Recall@150': 0.0, 'Recall@200': 0.0}
    Collecting QA prompts: 100%|███████████████████████████████████████████████████| 17/17 [00:00<00:00, 6282.22it/s]
    QA Reading:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:   6%|███▊                                                             | 1/17 [00:01<00:26,  1.67s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:  12%|███████▋                                                         | 2/17 [00:02<00:21,  1.43s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:  18%|███████████▍                                                     | 3/17 [00:04<00:17,  1.28s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:  24%|███████████████▎                                                 | 4/17 [00:05<00:17,  1.38s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:  35%|██████████████████████▉                                          | 6/17 [00:06<00:10,  1.07it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:  41%|██████████████████████████▊                                      | 7/17 [00:08<00:11,  1.18s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:  47%|██████████████████████████████▌                                  | 8/17 [00:09<00:11,  1.23s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:  53%|██████████████████████████████████▍                              | 9/17 [00:10<00:09,  1.19s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:  59%|█████████████████████████████████████▋                          | 10/17 [00:11<00:07,  1.11s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:  65%|█████████████████████████████████████████▍                      | 11/17 [00:12<00:06,  1.04s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:  82%|████████████████████████████████████████████████████▋           | 14/17 [00:13<00:02,  1.48it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:  88%|████████████████████████████████████████████████████████▍       | 15/17 [00:14<00:01,  1.32it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:  94%|████████████████████████████████████████████████████████████▏   | 16/17 [00:16<00:00,  1.21it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:17<00:00,  1.02s/it]
    Extraction Answers from LLM Response: 17it [00:00, 315500.74it/s]
    INFO:src.hipporag.HippoRAG:Evaluation results for QA: {'ExactMatch': 0.2941, 'F1': 0.6663}
  ```

- 0.3

  ```txt
  (hipporag) perrywu12@instance-20250323-150245:~/HippoRAG$ python main.py --dataset itsupport --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2
    INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/perrywu12/HippoRAG/src/hipporag/prompts/templates
    INFO:src.hipporag.HippoRAG:Loaded graph from outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/graph.pickle with 1398 nodes, 9045 edges
    INFO:datasets:PyTorch version 2.5.1 available.
    INFO:datasets:Polars version 1.26.0 available.
    INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
    Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 4/4 [00:05<00:00,  1.29s/it]
    INFO:src.hipporag.embedding_store:Loaded 100 records from outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet
    INFO:src.hipporag.embedding_store:Loaded 1298 records from outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/entity_embeddings/vdb_entity.parquet
    INFO:src.hipporag.embedding_store:Loaded 1456 records from outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/fact_embeddings/vdb_fact.parquet
    INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/perrywu12/HippoRAG/src/hipporag/prompts/templates
    INFO:src.hipporag.HippoRAG:Indexing Documents
    INFO:src.hipporag.HippoRAG:Performing OpenIE
    INFO:src.hipporag.embedding_store:Inserting 0 new records, 100 records already exist.
    INFO:src.hipporag.HippoRAG:OpenIE results saved to outputs/itsupport/openie_results_ner_gpt-4o-mini.json
    INFO:src.hipporag.HippoRAG:Encoding Entities
    INFO:src.hipporag.embedding_store:Inserting 0 new records, 1298 records already exist.
    INFO:src.hipporag.HippoRAG:Encoding Facts
    INFO:src.hipporag.embedding_store:Inserting 0 new records, 1456 records already exist.
    INFO:src.hipporag.HippoRAG:Constructing Graph
    INFO:src.hipporag.HippoRAG:Adding OpenIE triples to graph.
    100it [00:00, 16960.39it/s]
    INFO:src.hipporag.HippoRAG:Connecting passage nodes to phrase nodes.
    100it [00:00, 1784810.21it/s]
    INFO:src.hipporag.HippoRAG:Preparing for fast retrieval.
    INFO:src.hipporag.HippoRAG:Loading keys.
    INFO:src.hipporag.HippoRAG:Loading embeddings.
    INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_fact.
    Batch Encoding:   0%|                                                                     | 0/17 [00:00<?, ?it/s]/home/perrywu12/.cache/huggingface/modules/transformers_modules/nvidia/NV-Embed-v2/c50d55f43bde7e6a18e0eaa15a62fd63a930f1a1/modeling_nvembed.py:349: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
    'input_ids': torch.tensor(batch_dict.get('input_ids').to(batch_dict.get('input_ids')).long()),
    /home/perrywu12/miniconda3/envs/hipporag/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
    self.gen = func(*args, **kwds)
    Batch Encoding: 24it [00:00, 24.80it/s]                                                                          
    INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_passage.
    Batch Encoding: 24it [00:00, 49.55it/s]                                                                          
    Retrieving: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:01<00:00,  8.90it/s]
    INFO:src.hipporag.HippoRAG:Total Retrieval Time 3.47s
    INFO:src.hipporag.HippoRAG:Total Recognition Memory Time 0.28s
    INFO:src.hipporag.HippoRAG:Total PPR Time 1.55s
    INFO:src.hipporag.HippoRAG:Total Misc Time 1.65s
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    INFO:src.hipporag.HippoRAG:Evaluation results for retrieval: {'Recall@1': 0.0, 'Recall@2': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0, 'Recall@20': 0.0, 'Recall@30': 0.0, 'Recall@50': 0.0, 'Recall@100': 0.0, 'Recall@150': 0.0, 'Recall@200': 0.0}
    Collecting QA prompts: 100%|███████████████████████████████████████████████████| 17/17 [00:00<00:00, 7186.37it/s]
    QA Reading:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:   6%|███▊                                                             | 1/17 [00:01<00:30,  1.88s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:  35%|██████████████████████▉                                          | 6/17 [00:03<00:06,  1.81it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:  47%|██████████████████████████████▌                                  | 8/17 [00:05<00:05,  1.53it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading:  59%|█████████████████████████████████████▋                          | 10/17 [00:06<00:04,  1.75it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:07<00:00,  2.38it/s]
    Extraction Answers from LLM Response: 17it [00:00, 249311.78it/s]
    INFO:src.hipporag.HippoRAG:Evaluation results for QA: {'ExactMatch': 0.2941, 'F1': 0.6663}
  ```
- 0.1 
- 0.05

  ```txt
    (hipporag) perrywu12@instance-20250323-150245:~/HippoRAG$ python main.py --dataset itsupport --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2
    INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/perrywu12/HippoRAG/src/hipporag/prompts/templates
    INFO:datasets:PyTorch version 2.5.1 available.
    INFO:datasets:Polars version 1.26.0 available.
    INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
    Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 4/4 [01:29<00:00, 22.48s/it]
    INFO:src.hipporag.embedding_store:Loaded 100 records from outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet
    INFO:src.hipporag.embedding_store:Loaded 1298 records from outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/entity_embeddings/vdb_entity.parquet
    INFO:src.hipporag.embedding_store:Loaded 1456 records from outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/fact_embeddings/vdb_fact.parquet
    INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/perrywu12/HippoRAG/src/hipporag/prompts/templates
    INFO:src.hipporag.HippoRAG:Indexing Documents
    INFO:src.hipporag.HippoRAG:Performing OpenIE
    INFO:src.hipporag.embedding_store:Inserting 0 new records, 100 records already exist.
    INFO:src.hipporag.HippoRAG:OpenIE results saved to outputs/itsupport/openie_results_ner_gpt-4o-mini.json
    INFO:src.hipporag.HippoRAG:Encoding Entities
    INFO:src.hipporag.embedding_store:Inserting 0 new records, 1298 records already exist.
    INFO:src.hipporag.HippoRAG:Encoding Facts
    INFO:src.hipporag.embedding_store:Inserting 0 new records, 1456 records already exist.
    INFO:src.hipporag.HippoRAG:Constructing Graph
    INFO:src.hipporag.HippoRAG:Adding OpenIE triples to graph.
    100it [00:00, 17179.21it/s]
    INFO:src.hipporag.HippoRAG:Connecting passage nodes to phrase nodes.
    100it [00:00, 45620.01it/s]
    INFO:src.hipporag.HippoRAG:Found 100 new chunks to save into graph.
    INFO:src.hipporag.HippoRAG:Expanding graph with synonymy edges
    INFO:src.hipporag.HippoRAG:Performing KNN retrieval for each phrase nodes (1298).
    KNN for Queries: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  2.93it/s]
    100%|█████████████████████████████████████████████████████████████████████| 1298/1298 [00:00<00:00, 89575.28it/s]
    INFO:src.hipporag.HippoRAG:Graph construction completed!
    {'num_phrase_nodes': 1298, 'num_passage_nodes': 100, 'num_total_nodes': 1398, 'num_extracted_triples': 1456, 'num_triples_with_passage_node': 1832, 'num_synonymy_triples': 5760, 'num_total_triples': 9048}
    INFO:src.hipporag.HippoRAG:Writing graph with 1398 nodes, 9045 edges
    INFO:src.hipporag.HippoRAG:Saving graph completed!
    INFO:src.hipporag.HippoRAG:Preparing for fast retrieval.
    INFO:src.hipporag.HippoRAG:Loading keys.
    INFO:src.hipporag.HippoRAG:Loading embeddings.
    INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_fact.
    Batch Encoding:   0%|                                                                     | 0/17 [00:00<?, ?it/s]/home/perrywu12/.cache/huggingface/modules/transformers_modules/nvidia/NV-Embed-v2/c50d55f43bde7e6a18e0eaa15a62fd63a930f1a1/modeling_nvembed.py:349: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
    'input_ids': torch.tensor(batch_dict.get('input_ids').to(batch_dict.get('input_ids')).long()),
    /home/perrywu12/miniconda3/envs/hipporag/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
    self.gen = func(*args, **kwds)
    Batch Encoding: 24it [00:01, 20.68it/s]                                                                          
    INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_passage.
    Batch Encoding: 24it [00:00, 48.96it/s]                                                                          
    Retrieving: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:02<00:00,  7.04it/s]
    INFO:src.hipporag.HippoRAG:Total Retrieval Time 4.17s
    INFO:src.hipporag.HippoRAG:Total Recognition Memory Time 0.29s
    INFO:src.hipporag.HippoRAG:Total PPR Time 2.04s
    INFO:src.hipporag.HippoRAG:Total Misc Time 1.84s
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
    INFO:src.hipporag.HippoRAG:Evaluation results for retrieval: {'Recall@1': 0.0, 'Recall@2': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0, 'Recall@20': 0.0, 'Recall@30': 0.0, 'Recall@50': 0.0, 'Recall@100': 0.0, 'Recall@150': 0.0, 'Recall@200': 0.0}
    Collecting QA prompts: 100%|███████████████████████████████████████████████████| 17/17 [00:00<00:00, 6298.87it/s]
    QA Reading:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    QA Reading: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:01<00:00,  9.46it/s]
    Extraction Answers from LLM Response: 17it [00:00, 349525.33it/s]
    INFO:src.hipporag.HippoRAG:Evaluation results for QA: {'ExactMatch': 0.3529, 'F1': 0.7025}
    ```
- 0.01 