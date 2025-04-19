# experimental results

## passage_node_weight

- 0.5


- 0.3
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