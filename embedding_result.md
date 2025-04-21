| embedding model name | results |
| ---- | ---- |
| GritLM/GritLM-7B | 'ExactMatch': 0.4118, 'F1': 0.7379 |
| nvidia/NV-Embed-v2 | 'ExactMatch': 0.3529, 'F1': 0.7114 |
| facebook/contriever | 'ExactMatch': 0.2941, 'F1': 0.695 |
| BAAI/bge-small-en-v1.5 | 'ExactMatch': 0.2941, 'F1': 0.6817 |
| Salesforce/SFR-Embedding-Mistral | 'ExactMatch': 0.2941, 'F1': 0.6066 |

```
(hipporag) carolsong1110@instance-20250416-075442:~/HippoRAG$ python main.py --dataset sample  --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name GritLM/GritLM-7B
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
INFO:src.hipporag.HippoRAG:Loaded graph from outputs/sample/gpt-4o-mini_GritLM_GritLM-7B/graph.pickle with 250 nodes, 1220 edges
INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 3/3 [00:04<00:00,  1.52s/it]
Created GritLM: torch.bfloat16 dtype, mean pool, unified mode, bbcc attn
INFO:src.hipporag.embedding_store:Loaded 14 records from outputs/sample/gpt-4o-mini_GritLM_GritLM-7B/chunk_embeddings/vdb_chunk.parquet
INFO:src.hipporag.embedding_store:Loaded 236 records from outputs/sample/gpt-4o-mini_GritLM_GritLM-7B/entity_embeddings/vdb_entity.parquet
INFO:src.hipporag.embedding_store:Loaded 210 records from outputs/sample/gpt-4o-mini_GritLM_GritLM-7B/fact_embeddings/vdb_fact.parquet
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
INFO:src.hipporag.HippoRAG:Indexing Documents
INFO:src.hipporag.HippoRAG:Performing OpenIE
INFO:src.hipporag.embedding_store:Inserting 89 new records, 11 records already exist.
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 103 records to outputs/sample/gpt-4o-mini_GritLM_GritLM-7B/chunk_embeddings/vdb_chunk.parquet
NER:   0%|                                                                                | 0/89 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
NER: 100%|█| 89/89 [00:09<00:00,  9.52it/s, total_prompt_tokens=61747, total_completion_tokens=4205, num_cache_hi
Extracting triples:   0%|                                                                 | 0/89 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

Extracting triples: 100%|█| 89/89 [00:21<00:00,  4.12it/s, total_prompt_tokens=90917, total_completion_tokens=192
INFO:src.hipporag.HippoRAG:OpenIE results saved to outputs/sample/openie_results_ner_gpt-4o-mini.json
INFO:src.hipporag.HippoRAG:Encoding Entities
INFO:src.hipporag.embedding_store:Inserting 1105 new records, 236 records already exist.
Batches: 100%|█████████████████████████████████████████████████████████████████| 139/139 [00:10<00:00, 13.86it/s]
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1341 records to outputs/sample/gpt-4o-mini_GritLM_GritLM-7B/entity_embeddings/vdb_entity.parquet
INFO:src.hipporag.HippoRAG:Encoding Facts
INFO:src.hipporag.embedding_store:Inserting 1260 new records, 210 records already exist.
Batches: 100%|█████████████████████████████████████████████████████████████████| 158/158 [00:13<00:00, 11.30it/s]
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1470 records to outputs/sample/gpt-4o-mini_GritLM_GritLM-7B/fact_embeddings/vdb_fact.parquet
INFO:src.hipporag.HippoRAG:Constructing Graph
INFO:src.hipporag.HippoRAG:Adding OpenIE triples to graph.
103it [00:00, 16027.80it/s]
INFO:src.hipporag.HippoRAG:Connecting passage nodes to phrase nodes.
103it [00:00, 41444.10it/s]
INFO:src.hipporag.HippoRAG:Found 89 new chunks to save into graph.
INFO:src.hipporag.HippoRAG:Expanding graph with synonymy edges
INFO:src.hipporag.HippoRAG:Performing KNN retrieval for each phrase nodes (1341).
KNN for Queries: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.60it/s]
100%|█████████████████████████████████████████████████████████████████████| 1341/1341 [00:00<00:00, 35838.02it/s]
INFO:src.hipporag.HippoRAG:Graph construction completed!
{'num_phrase_nodes': 1341, 'num_passage_nodes': 103, 'num_total_nodes': 1444, 'num_extracted_triples': 1470, 'num_triples_with_passage_node': 1607, 'num_synonymy_triples': 8578, 'num_total_triples': 11655}
INFO:src.hipporag.HippoRAG:Writing graph with 1444 nodes, 12872 edges
INFO:src.hipporag.HippoRAG:Saving graph completed!
INFO:src.hipporag.HippoRAG:Preparing for fast retrieval.
INFO:src.hipporag.HippoRAG:Loading keys.
INFO:src.hipporag.HippoRAG:Loading embeddings.
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_fact.
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_passage.
Retrieving:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  94%|████████████████████████████████████████████████████████████▏   | 16/17 [00:19<00:01,  1.37s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:20<00:00,  1.21s/it]
INFO:src.hipporag.HippoRAG:Total Retrieval Time 21.36s
INFO:src.hipporag.HippoRAG:Total Recognition Memory Time 20.02s
INFO:src.hipporag.HippoRAG:Total PPR Time 0.42s
INFO:src.hipporag.HippoRAG:Total Misc Time 0.92s
WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (103) is smaller than largest topk for recall score (200)
WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (103) is smaller than largest topk for recall score (200)
INFO:src.hipporag.HippoRAG:Evaluation results for retrieval: {'Recall@1': 0.0, 'Recall@2': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0, 'Recall@20': 0.0, 'Recall@30': 0.0, 'Recall@50': 0.0, 'Recall@100': 0.0, 'Recall@150': 0.0, 'Recall@200': 0.0}
Collecting QA prompts: 100%|███████████████████████████████████████████████████| 17/17 [00:00<00:00, 2550.91it/s]
QA Reading:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  94%|████████████████████████████████████████████████████████████▏   | 16/17 [00:27<00:01,  1.54s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:29<00:00,  1.71s/it]
Extraction Answers from LLM Response: 17it [00:00, 339538.90it/s]
INFO:src.hipporag.HippoRAG:Evaluation results for QA: {'ExactMatch': 0.4118, 'F1': 0.7379}
```

```
(hipporag) carolsong1110@instance-20250416-075442:~/HippoRAG$ python main.py --dataset sample  --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name facebook/contriever
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
INFO:src.hipporag.HippoRAG:Loaded graph from outputs/sample/gpt-4o-mini_facebook_contriever/graph.pickle with 37 nodes, 110 edges
INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
INFO:src.hipporag.embedding_store:Loaded 3 records from outputs/sample/gpt-4o-mini_facebook_contriever/chunk_embeddings/vdb_chunk.parquet
INFO:src.hipporag.embedding_store:Loaded 34 records from outputs/sample/gpt-4o-mini_facebook_contriever/entity_embeddings/vdb_entity.parquet
INFO:src.hipporag.embedding_store:Loaded 33 records from outputs/sample/gpt-4o-mini_facebook_contriever/fact_embeddings/vdb_fact.parquet
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
INFO:src.hipporag.HippoRAG:Indexing Documents
INFO:src.hipporag.HippoRAG:Performing OpenIE
INFO:src.hipporag.embedding_store:Inserting 100 new records, 0 records already exist.
Batch Encoding: 104it [00:01, 71.68it/s]                                                                         
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 103 records to outputs/sample/gpt-4o-mini_facebook_contriever/chunk_embeddings/vdb_chunk.parquet
INFO:src.hipporag.HippoRAG:OpenIE results saved to outputs/sample/openie_results_ner_gpt-4o-mini.json
INFO:src.hipporag.HippoRAG:Encoding Entities
INFO:src.hipporag.embedding_store:Inserting 1307 new records, 34 records already exist.
Batch Encoding: 1312it [00:01, 856.92it/s]                                                                       
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1341 records to outputs/sample/gpt-4o-mini_facebook_contriever/entity_embeddings/vdb_entity.parquet
INFO:src.hipporag.HippoRAG:Encoding Facts
INFO:src.hipporag.embedding_store:Inserting 1437 new records, 33 records already exist.
Batch Encoding: 1440it [00:01, 814.17it/s]                                                                       
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1470 records to outputs/sample/gpt-4o-mini_facebook_contriever/fact_embeddings/vdb_fact.parquet
INFO:src.hipporag.HippoRAG:Constructing Graph
INFO:src.hipporag.HippoRAG:Adding OpenIE triples to graph.
103it [00:00, 16528.17it/s]
INFO:src.hipporag.HippoRAG:Connecting passage nodes to phrase nodes.
103it [00:00, 45020.15it/s]
INFO:src.hipporag.HippoRAG:Found 100 new chunks to save into graph.
INFO:src.hipporag.HippoRAG:Expanding graph with synonymy edges
INFO:src.hipporag.HippoRAG:Performing KNN retrieval for each phrase nodes (1341).
KNN for Queries: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  4.28it/s]
100%|█████████████████████████████████████████████████████████████████████| 1341/1341 [00:00<00:00, 43884.97it/s]
INFO:src.hipporag.HippoRAG:Graph construction completed!
{'num_phrase_nodes': 1341, 'num_passage_nodes': 103, 'num_total_nodes': 1444, 'num_extracted_triples': 1470, 'num_triples_with_passage_node': 1831, 'num_synonymy_triples': 2403, 'num_total_triples': 5704}
INFO:src.hipporag.HippoRAG:Writing graph with 1444 nodes, 5811 edges
INFO:src.hipporag.HippoRAG:Saving graph completed!
INFO:src.hipporag.HippoRAG:Preparing for fast retrieval.
INFO:src.hipporag.HippoRAG:Loading keys.
INFO:src.hipporag.HippoRAG:Loading embeddings.
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_fact.
Batch Encoding: 24it [00:00, 523.58it/s]                                                                         
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_passage.
Batch Encoding: 24it [00:00, 834.77it/s]                                                                         
Retrieving:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:24<00:00,  1.46s/it]
INFO:src.hipporag.HippoRAG:Total Retrieval Time 24.92s
INFO:src.hipporag.HippoRAG:Total Recognition Memory Time 24.49s
INFO:src.hipporag.HippoRAG:Total PPR Time 0.27s
INFO:src.hipporag.HippoRAG:Total Misc Time 0.16s
WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (103) is smaller than largest topk for recall score (200)
WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (103) is smaller than largest topk for recall score (200)
INFO:src.hipporag.HippoRAG:Evaluation results for retrieval: {'Recall@1': 0.0, 'Recall@2': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0, 'Recall@20': 0.0, 'Recall@30': 0.0, 'Recall@50': 0.0, 'Recall@100': 0.0, 'Recall@150': 0.0, 'Recall@200': 0.0}
Collecting QA prompts: 100%|███████████████████████████████████████████████████| 17/17 [00:00<00:00, 6369.20it/s]
QA Reading:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  94%|████████████████████████████████████████████████████████████▏   | 16/17 [00:25<00:01,  1.49s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:26<00:00,  1.58s/it]
Extraction Answers from LLM Response: 17it [00:00, 286358.10it/s]
INFO:src.hipporag.HippoRAG:Evaluation results for QA: {'ExactMatch': 0.2941, 'F1': 0.695}
```

```
(hipporag) carolsong1110@instance-20250416-075442:~/HippoRAG$ python main.py --dataset sample  --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name BAAI/bge-small-en-v1.5
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
INFO:src.hipporag.HippoRAG:Loaded graph from outputs/sample/gpt-4o-mini_BAAI_bge-small-en-v1.5/graph.pickle with 37 nodes, 110 edges
INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
INFO:src.hipporag.embedding_store:Loaded 3 records from outputs/sample/gpt-4o-mini_BAAI_bge-small-en-v1.5/chunk_embeddings/vdb_chunk.parquet
INFO:src.hipporag.embedding_store:Loaded 34 records from outputs/sample/gpt-4o-mini_BAAI_bge-small-en-v1.5/entity_embeddings/vdb_entity.parquet
INFO:src.hipporag.embedding_store:Loaded 33 records from outputs/sample/gpt-4o-mini_BAAI_bge-small-en-v1.5/fact_embeddings/vdb_fact.parquet
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
INFO:src.hipporag.HippoRAG:Indexing Documents
INFO:src.hipporag.HippoRAG:Performing OpenIE
INFO:src.hipporag.embedding_store:Inserting 100 new records, 0 records already exist.
Batch Encoding: 104it [00:00, 159.15it/s]                                                                        
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 103 records to outputs/sample/gpt-4o-mini_BAAI_bge-small-en-v1.5/chunk_embeddings/vdb_chunk.parquet
INFO:src.hipporag.HippoRAG:OpenIE results saved to outputs/sample/openie_results_ner_gpt-4o-mini.json
INFO:src.hipporag.HippoRAG:Encoding Entities
INFO:src.hipporag.embedding_store:Inserting 1307 new records, 34 records already exist.
Batch Encoding: 1312it [00:01, 853.71it/s]                                                                       
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1341 records to outputs/sample/gpt-4o-mini_BAAI_bge-small-en-v1.5/entity_embeddings/vdb_entity.parquet
INFO:src.hipporag.HippoRAG:Encoding Facts
INFO:src.hipporag.embedding_store:Inserting 1437 new records, 33 records already exist.
Batch Encoding: 1440it [00:01, 830.34it/s]                                                                       
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1470 records to outputs/sample/gpt-4o-mini_BAAI_bge-small-en-v1.5/fact_embeddings/vdb_fact.parquet
INFO:src.hipporag.HippoRAG:Constructing Graph
INFO:src.hipporag.HippoRAG:Adding OpenIE triples to graph.
103it [00:00, 17174.74it/s]
INFO:src.hipporag.HippoRAG:Connecting passage nodes to phrase nodes.
103it [00:00, 43841.42it/s]
INFO:src.hipporag.HippoRAG:Found 100 new chunks to save into graph.
INFO:src.hipporag.HippoRAG:Expanding graph with synonymy edges
INFO:src.hipporag.HippoRAG:Performing KNN retrieval for each phrase nodes (1341).
KNN for Queries: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  4.45it/s]
100%|█████████████████████████████████████████████████████████████████████| 1341/1341 [00:00<00:00, 42802.28it/s]
INFO:src.hipporag.HippoRAG:Graph construction completed!
{'num_phrase_nodes': 1341, 'num_passage_nodes': 103, 'num_total_nodes': 1444, 'num_extracted_triples': 1470, 'num_triples_with_passage_node': 1831, 'num_synonymy_triples': 4417, 'num_total_triples': 7718}
INFO:src.hipporag.HippoRAG:Writing graph with 1444 nodes, 7825 edges
INFO:src.hipporag.HippoRAG:Saving graph completed!
INFO:src.hipporag.HippoRAG:Preparing for fast retrieval.
INFO:src.hipporag.HippoRAG:Loading keys.
INFO:src.hipporag.HippoRAG:Loading embeddings.
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_fact.
Batch Encoding: 24it [00:00, 724.98it/s]                                                                         
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_passage.
Batch Encoding: 24it [00:00, 882.39it/s]                                                                         
Retrieving:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  94%|████████████████████████████████████████████████████████████▏   | 16/17 [00:22<00:01,  1.27s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:24<00:00,  1.44s/it]
INFO:src.hipporag.HippoRAG:Total Retrieval Time 24.49s
INFO:src.hipporag.HippoRAG:Total Recognition Memory Time 24.08s
INFO:src.hipporag.HippoRAG:Total PPR Time 0.27s
INFO:src.hipporag.HippoRAG:Total Misc Time 0.14s
WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (103) is smaller than largest topk for recall score (200)
WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (103) is smaller than largest topk for recall score (200)
INFO:src.hipporag.HippoRAG:Evaluation results for retrieval: {'Recall@1': 0.0, 'Recall@2': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0, 'Recall@20': 0.0, 'Recall@30': 0.0, 'Recall@50': 0.0, 'Recall@100': 0.0, 'Recall@150': 0.0, 'Recall@200': 0.0}
Collecting QA prompts: 100%|███████████████████████████████████████████████████| 17/17 [00:00<00:00, 2595.11it/s]
QA Reading:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  94%|████████████████████████████████████████████████████████████▏   | 16/17 [00:28<00:01,  1.62s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:30<00:00,  1.80s/it]
Extraction Answers from LLM Response: 17it [00:00, 241705.65it/s]
INFO:src.hipporag.HippoRAG:Evaluation results for QA: {'ExactMatch': 0.2941, 'F1': 0.6817}
```

```
(hipporag) carolsong1110@instance-20250416-075442:~/HippoRAG$ python main.py --dataset sample  --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
INFO:src.hipporag.HippoRAG:Loaded graph from outputs/sample/gpt-4o-mini_nvidia_NV-Embed-v2/graph.pickle with 37 nodes, 118 edges
INFO:datasets:PyTorch version 2.5.1 available.
INFO:datasets:Polars version 1.27.1 available.
INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 4/4 [00:05<00:00,  1.29s/it]
INFO:src.hipporag.embedding_store:Loaded 3 records from outputs/sample/gpt-4o-mini_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet
INFO:src.hipporag.embedding_store:Loaded 34 records from outputs/sample/gpt-4o-mini_nvidia_NV-Embed-v2/entity_embeddings/vdb_entity.parquet
INFO:src.hipporag.embedding_store:Loaded 33 records from outputs/sample/gpt-4o-mini_nvidia_NV-Embed-v2/fact_embeddings/vdb_fact.parquet
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
INFO:src.hipporag.HippoRAG:Indexing Documents
INFO:src.hipporag.HippoRAG:Performing OpenIE
INFO:src.hipporag.embedding_store:Inserting 100 new records, 0 records already exist.
Batch Encoding:   0%|                                                                    | 0/100 [00:00<?, ?it/s]/home/carolsong1110/.cache/huggingface/modules/transformers_modules/nvidia/NV-Embed-v2/c50d55f43bde7e6a18e0eaa15a62fd63a930f1a1/modeling_nvembed.py:349: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  'input_ids': torch.tensor(batch_dict.get('input_ids').to(batch_dict.get('input_ids')).long()),
/home/carolsong1110/miniconda3/envs/hipporag/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
Batch Encoding: 104it [00:29,  3.51it/s]                                                                         
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 103 records to outputs/sample/gpt-4o-mini_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet
INFO:src.hipporag.HippoRAG:OpenIE results saved to outputs/sample/openie_results_ner_gpt-4o-mini.json
INFO:src.hipporag.HippoRAG:Encoding Entities
INFO:src.hipporag.embedding_store:Inserting 1307 new records, 34 records already exist.
Batch Encoding: 1312it [00:21, 61.72it/s]                                                                        
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1341 records to outputs/sample/gpt-4o-mini_nvidia_NV-Embed-v2/entity_embeddings/vdb_entity.parquet
INFO:src.hipporag.HippoRAG:Encoding Facts
INFO:src.hipporag.embedding_store:Inserting 1437 new records, 33 records already exist.
Batch Encoding: 1440it [00:30, 46.86it/s]                                                                        
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1470 records to outputs/sample/gpt-4o-mini_nvidia_NV-Embed-v2/fact_embeddings/vdb_fact.parquet
INFO:src.hipporag.HippoRAG:Constructing Graph
INFO:src.hipporag.HippoRAG:Adding OpenIE triples to graph.
103it [00:00, 14516.09it/s]
INFO:src.hipporag.HippoRAG:Connecting passage nodes to phrase nodes.
103it [00:00, 36160.82it/s]
INFO:src.hipporag.HippoRAG:Found 100 new chunks to save into graph.
INFO:src.hipporag.HippoRAG:Expanding graph with synonymy edges
INFO:src.hipporag.HippoRAG:Performing KNN retrieval for each phrase nodes (1341).
KNN for Queries: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.14it/s]
100%|█████████████████████████████████████████████████████████████████████| 1341/1341 [00:00<00:00, 41451.86it/s]
INFO:src.hipporag.HippoRAG:Graph construction completed!
{'num_phrase_nodes': 1341, 'num_passage_nodes': 103, 'num_total_nodes': 1444, 'num_extracted_triples': 1470, 'num_triples_with_passage_node': 1831, 'num_synonymy_triples': 6103, 'num_total_triples': 9404}
INFO:src.hipporag.HippoRAG:Writing graph with 1444 nodes, 9519 edges
INFO:src.hipporag.HippoRAG:Saving graph completed!
INFO:src.hipporag.HippoRAG:Preparing for fast retrieval.
INFO:src.hipporag.HippoRAG:Loading keys.
INFO:src.hipporag.HippoRAG:Loading embeddings.
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_fact.
Batch Encoding: 24it [00:00, 45.38it/s]                                                                          
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_passage.
Batch Encoding: 24it [00:00, 42.80it/s]                                                                          
Retrieving:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:27<00:00,  1.61s/it]
INFO:src.hipporag.HippoRAG:Total Retrieval Time 28.55s
INFO:src.hipporag.HippoRAG:Total Recognition Memory Time 27.17s
INFO:src.hipporag.HippoRAG:Total PPR Time 0.07s
INFO:src.hipporag.HippoRAG:Total Misc Time 1.31s
WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (103) is smaller than largest topk for recall score (200)
WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (103) is smaller than largest topk for recall score (200)
INFO:src.hipporag.HippoRAG:Evaluation results for retrieval: {'Recall@1': 0.0, 'Recall@2': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0, 'Recall@20': 0.0, 'Recall@30': 0.0, 'Recall@50': 0.0, 'Recall@100': 0.0, 'Recall@150': 0.0, 'Recall@200': 0.0}
Collecting QA prompts: 100%|███████████████████████████████████████████████████| 17/17 [00:00<00:00, 2530.10it/s]
QA Reading:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:24<00:00,  1.45s/it]
Extraction Answers from LLM Response: 17it [00:00, 312733.19it/s]
INFO:src.hipporag.HippoRAG:Evaluation results for QA: {'ExactMatch': 0.3529, 'F1': 0.7114}
```
```
(hipporag) carolsong1110@instance-20250416-075442:~/HippoRAG$ python main.py --dataset sample  --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name Salesforce/SFR-Embedding-Mistral
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
tokenizer_config.json: 100%|████████████████████████████████████████████████████| 981/981 [00:00<00:00, 7.12MB/s]
tokenizer.model: 100%|████████████████████████████████████████████████████████| 493k/493k [00:00<00:00, 29.8MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████| 1.80M/1.80M [00:00<00:00, 2.50MB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████| 624/624 [00:00<00:00, 4.84MB/s]
config.json: 100%|██████████████████████████████████████████████████████████████| 663/663 [00:00<00:00, 4.66MB/s]
model.safetensors.index.json: 100%|█████████████████████████████████████████| 22.2k/22.2k [00:00<00:00, 98.7MB/s]
model-00001-of-00003.safetensors: 100%|██████████████████████████████████████| 4.94G/4.94G [00:11<00:00, 417MB/s]
model-00002-of-00003.safetensors: 100%|██████████████████████████████████████| 5.00G/5.00G [00:12<00:00, 400MB/s]
model-00003-of-00003.safetensors: 100%|██████████████████████████████████████| 4.28G/4.28G [00:11<00:00, 385MB/s]
Downloading shards: 100%|██████████████████████████████████████████████████████████| 3/3 [00:36<00:00, 12.04s/it]
INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 3/3 [00:04<00:00,  1.54s/it]
INFO:src.hipporag.embedding_store:Creating working directory: outputs/sample/gpt-4o-mini_Salesforce_SFR-Embedding-Mistral/chunk_embeddings
INFO:src.hipporag.embedding_store:Creating working directory: outputs/sample/gpt-4o-mini_Salesforce_SFR-Embedding-Mistral/entity_embeddings
INFO:src.hipporag.embedding_store:Creating working directory: outputs/sample/gpt-4o-mini_Salesforce_SFR-Embedding-Mistral/fact_embeddings
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
INFO:src.hipporag.HippoRAG:Indexing Documents
INFO:src.hipporag.HippoRAG:Performing OpenIE
INFO:src.hipporag.embedding_store:Inserting 100 new records, 0 records already exist.
Batch Encoding:   0%|                                                                    | 0/100 [00:00<?, ?it/s]Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Batch Encoding: 104it [00:26,  3.96it/s]                                                                         
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 100 records to outputs/sample/gpt-4o-mini_Salesforce_SFR-Embedding-Mistral/chunk_embeddings/vdb_chunk.parquet
NER:   0%|                                                                               | 0/100 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
NER:   1%| | 1/100 [00:03<06:03,  3.67s/it, total_prompt_tokens=679, total_completion_tokens=25, num_cache_hit=0]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
NER:   2%| | 2/100 [00:05<03:54,  2.39s/it, total_prompt_tokens=1361, total_completion_tokens=66, num_cache_hit=0INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
NER: 100%|█| 100/100 [00:23<00:00,  4.18it/s, total_prompt_tokens=69379, total_completion_tokens=5045, num_cache_
Extracting triples:   0%|                                                                | 0/100 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Extracting triples: 100%|█| 100/100 [00:28<00:00,  3.56it/s, total_prompt_tokens=102374, total_completion_tokens=
INFO:src.hipporag.HippoRAG:OpenIE results saved to outputs/sample/openie_results_ner_gpt-4o-mini.json
INFO:src.hipporag.HippoRAG:Encoding Entities
INFO:src.hipporag.embedding_store:Inserting 1260 new records, 0 records already exist.
Batch Encoding: 1264it [00:11, 109.67it/s]                                                                       
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1260 records to outputs/sample/gpt-4o-mini_Salesforce_SFR-Embedding-Mistral/entity_embeddings/vdb_entity.parquet
INFO:src.hipporag.HippoRAG:Encoding Facts
INFO:src.hipporag.embedding_store:Inserting 1430 new records, 0 records already exist.
Batch Encoding: 1432it [00:16, 85.44it/s]                                                                        
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1430 records to outputs/sample/gpt-4o-mini_Salesforce_SFR-Embedding-Mistral/fact_embeddings/vdb_fact.parquet
INFO:src.hipporag.HippoRAG:Constructing Graph
INFO:src.hipporag.HippoRAG:Adding OpenIE triples to graph.
100it [00:00, 16681.13it/s]
INFO:src.hipporag.HippoRAG:Connecting passage nodes to phrase nodes.
100it [00:00, 38575.41it/s]
INFO:src.hipporag.HippoRAG:Found 100 new chunks to save into graph.
INFO:src.hipporag.HippoRAG:Expanding graph with synonymy edges
INFO:src.hipporag.HippoRAG:Performing KNN retrieval for each phrase nodes (1260).
KNN for Queries: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.98it/s]
100%|█████████████████████████████████████████████████████████████████████| 1260/1260 [00:00<00:00, 40476.26it/s]
INFO:src.hipporag.HippoRAG:Graph construction completed!
{'num_phrase_nodes': 1260, 'num_passage_nodes': 100, 'num_total_nodes': 1360, 'num_extracted_triples': 1430, 'num_triples_with_passage_node': 1781, 'num_synonymy_triples': 7355, 'num_total_triples': 10566}
INFO:src.hipporag.HippoRAG:Writing graph with 1360 nodes, 10566 edges
INFO:src.hipporag.HippoRAG:Saving graph completed!
INFO:src.hipporag.HippoRAG:Preparing for fast retrieval.
INFO:src.hipporag.HippoRAG:Loading keys.
INFO:src.hipporag.HippoRAG:Loading embeddings.
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_fact.
Batch Encoding: 24it [00:00, 84.66it/s]                                                                          
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_passage.
Batch Encoding: 24it [00:00, 82.41it/s]                                                                          
Retrieving:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:20<00:00,  1.19s/it]
INFO:src.hipporag.HippoRAG:Total Retrieval Time 20.92s
INFO:src.hipporag.HippoRAG:Total Recognition Memory Time 20.05s
INFO:src.hipporag.HippoRAG:Total PPR Time 0.15s
INFO:src.hipporag.HippoRAG:Total Misc Time 0.72s
WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
INFO:src.hipporag.HippoRAG:Evaluation results for retrieval: {'Recall@1': 0.0, 'Recall@2': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0, 'Recall@20': 0.0, 'Recall@30': 0.0, 'Recall@50': 0.0, 'Recall@100': 0.0, 'Recall@150': 0.0, 'Recall@200': 0.0}
Collecting QA prompts: 100%|███████████████████████████████████████████████████| 17/17 [00:00<00:00, 6458.04it/s]
QA Reading:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:25<00:00,  1.47s/it]
Extraction Answers from LLM Response: 17it [00:00, 311367.55it/s]
INFO:src.hipporag.HippoRAG:Evaluation results for QA: {'ExactMatch': 0.2941, 'F1': 0.6066}
```