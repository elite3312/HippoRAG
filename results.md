# results

- GritLM/GritLM-7B
```txt
(hipporag) perrywu12@instance-20250323-150245:~/HippoRAG$ python main.py --dataset itsupport --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name GritLM/GritLM-7B
INFO:src.hipporag.HippoRAG:Creating working directory: outputs/itsupport/gpt-4o-mini_GritLM_GritLM-7B
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/perrywu12/HippoRAG/src/hipporag/prompts/templates
config.json: 100%|██████████████████████████████████████████████████████████████| 946/946 [00:00<00:00, 5.07MB/s]
modeling_gritlm7b.py: 100%|█████████████████████████████████████████████████| 65.2k/65.2k [00:00<00:00, 14.6MB/s]
A new version of the following files was downloaded from https://huggingface.co/GritLM/GritLM-7B:
- modeling_gritlm7b.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
model.safetensors.index.json: 100%|█████████████████████████████████████████| 23.9k/23.9k [00:00<00:00, 99.9MB/s]
model-00001-of-00003.safetensors: 100%|██████████████████████████████████████| 4.94G/4.94G [00:18<00:00, 270MB/s]
model-00002-of-00003.safetensors: 100%|██████████████████████████████████████| 5.00G/5.00G [00:17<00:00, 285MB/s]
model-00003-of-00003.safetensors: 100%|██████████████████████████████████████| 4.54G/4.54G [00:15<00:00, 287MB/s]
Downloading shards: 100%|██████████████████████████████████████████████████████████| 3/3 [00:52<00:00, 17.45s/it]
INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 3/3 [00:04<00:00,  1.53s/it]
generation_config.json: 100%|████████████████████████████████████████████████████| 111/111 [00:00<00:00, 753kB/s]
Created GritLM: torch.bfloat16 dtype, mean pool, unified mode, bbcc attn
tokenizer_config.json: 100%|████████████████████████████████████████████████| 1.35k/1.35k [00:00<00:00, 10.3MB/s]
tokenizer.model: 100%|█████████████████████████████████████████████████████████| 493k/493k [00:00<00:00, 240MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████| 1.80M/1.80M [00:00<00:00, 2.55MB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████| 436/436 [00:00<00:00, 3.34MB/s]
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_GritLM_GritLM-7B/chunk_embeddings
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_GritLM_GritLM-7B/entity_embeddings
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_GritLM_GritLM-7B/fact_embeddings
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/perrywu12/HippoRAG/src/hipporag/prompts/templates
INFO:src.hipporag.HippoRAG:Indexing Documents
INFO:src.hipporag.HippoRAG:Performing OpenIE
INFO:src.hipporag.embedding_store:Inserting 100 new records, 0 records already exist.
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 100 records to outputs/itsupport/gpt-4o-mini_GritLM_GritLM-7B/chunk_embeddings/vdb_chunk.parquet
INFO:src.hipporag.HippoRAG:OpenIE results saved to outputs/itsupport/openie_results_ner_gpt-4o-mini.json
INFO:src.hipporag.HippoRAG:Encoding Entities
INFO:src.hipporag.embedding_store:Inserting 1298 new records, 0 records already exist.
Batches: 100%|█████████████████████████████████████████████████████████████████| 163/163 [00:11<00:00, 13.81it/s]
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1298 records to outputs/itsupport/gpt-4o-mini_GritLM_GritLM-7B/entity_embeddings/vdb_entity.parquet
INFO:src.hipporag.HippoRAG:Encoding Facts
INFO:src.hipporag.embedding_store:Inserting 1456 new records, 0 records already exist.
Batches: 100%|█████████████████████████████████████████████████████████████████| 182/182 [00:15<00:00, 11.39it/s]
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1456 records to outputs/itsupport/gpt-4o-mini_GritLM_GritLM-7B/fact_embeddings/vdb_fact.parquet
INFO:src.hipporag.HippoRAG:Constructing Graph
INFO:src.hipporag.HippoRAG:Adding OpenIE triples to graph.
100it [00:00, 16932.32it/s]
INFO:src.hipporag.HippoRAG:Connecting passage nodes to phrase nodes.
100it [00:00, 45236.24it/s]
INFO:src.hipporag.HippoRAG:Found 100 new chunks to save into graph.
INFO:src.hipporag.HippoRAG:Expanding graph with synonymy edges
INFO:src.hipporag.HippoRAG:Performing KNN retrieval for each phrase nodes (1298).
KNN for Queries: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.47it/s]
100%|█████████████████████████████████████████████████████████████████████| 1298/1298 [00:00<00:00, 55374.57it/s]
INFO:src.hipporag.HippoRAG:Graph construction completed!
{'num_phrase_nodes': 1298, 'num_passage_nodes': 100, 'num_total_nodes': 1398, 'num_extracted_triples': 1456, 'num_triples_with_passage_node': 1832, 'num_synonymy_triples': 9135, 'num_total_triples': 12423}
INFO:src.hipporag.HippoRAG:Writing graph with 1398 nodes, 12420 edges
INFO:src.hipporag.HippoRAG:Saving graph completed!
INFO:src.hipporag.HippoRAG:Preparing for fast retrieval.
INFO:src.hipporag.HippoRAG:Loading keys.
INFO:src.hipporag.HippoRAG:Loading embeddings.
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_fact.
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_passage.
Retrieving:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:   6%|███▊                                                             | 1/17 [00:02<00:38,  2.38s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  12%|███████▋                                                         | 2/17 [00:04<00:32,  2.16s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  18%|███████████▍                                                     | 3/17 [00:05<00:23,  1.71s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  24%|███████████████▎                                                 | 4/17 [00:07<00:21,  1.69s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  29%|███████████████████                                              | 5/17 [00:08<00:19,  1.63s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  35%|██████████████████████▉                                          | 6/17 [00:10<00:16,  1.52s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  41%|██████████████████████████▊                                      | 7/17 [00:11<00:14,  1.48s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  47%|██████████████████████████████▌                                  | 8/17 [00:13<00:14,  1.61s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  53%|██████████████████████████████████▍                              | 9/17 [00:15<00:13,  1.67s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  59%|█████████████████████████████████████▋                          | 10/17 [00:17<00:12,  1.81s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  65%|█████████████████████████████████████████▍                      | 11/17 [00:19<00:10,  1.82s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  71%|█████████████████████████████████████████████▏                  | 12/17 [00:20<00:08,  1.69s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  76%|████████████████████████████████████████████████▉               | 13/17 [00:21<00:06,  1.57s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  82%|████████████████████████████████████████████████████▋           | 14/17 [00:23<00:04,  1.54s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  88%|████████████████████████████████████████████████████████▍       | 15/17 [00:24<00:03,  1.51s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  94%|████████████████████████████████████████████████████████████▏   | 16/17 [00:26<00:01,  1.57s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:29<00:00,  1.73s/it]
INFO:src.hipporag.HippoRAG:Total Retrieval Time 30.24s
INFO:src.hipporag.HippoRAG:Total Recognition Memory Time 27.28s
INFO:src.hipporag.HippoRAG:Total PPR Time 2.06s
INFO:src.hipporag.HippoRAG:Total Misc Time 0.89s
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
Collecting QA prompts: 100%|███████████████████████████████████████████████████| 17/17 [00:00<00:00, 6253.57it/s]
QA Reading:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:   6%|███▊                                                             | 1/17 [00:01<00:23,  1.48s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  12%|███████▋                                                         | 2/17 [00:02<00:22,  1.49s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  24%|███████████████▎                                                 | 4/17 [00:04<00:14,  1.15s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  29%|███████████████████                                              | 5/17 [00:06<00:16,  1.37s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  41%|██████████████████████████▊                                      | 7/17 [00:09<00:14,  1.46s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  47%|██████████████████████████████▌                                  | 8/17 [00:11<00:13,  1.52s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  53%|██████████████████████████████████▍                              | 9/17 [00:12<00:11,  1.40s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  65%|█████████████████████████████████████████▍                      | 11/17 [00:14<00:06,  1.11s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  71%|█████████████████████████████████████████████▏                  | 12/17 [00:15<00:05,  1.11s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  76%|████████████████████████████████████████████████▉               | 13/17 [00:16<00:04,  1.21s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  82%|████████████████████████████████████████████████████▋           | 14/17 [00:18<00:03,  1.29s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  88%|████████████████████████████████████████████████████████▍       | 15/17 [00:19<00:02,  1.24s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  94%|████████████████████████████████████████████████████████████▏   | 16/17 [00:21<00:01,  1.38s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:22<00:00,  1.33s/it]
Extraction Answers from LLM Response: 17it [00:00, 342803.69it/s]
INFO:src.hipporag.HippoRAG:Evaluation results for QA: {'ExactMatch': 0.2941, 'F1': 0.6967}
```
- facebook/contriever 
```txt
(hipporag) perrywu12@instance-20250323-150245:~/HippoRAG$ python main.py --dataset itsupport --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name facebook/contriever 
INFO:src.hipporag.HippoRAG:Creating working directory: outputs/itsupport/gpt-4o-mini_facebook_contriever
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/perrywu12/HippoRAG/src/hipporag/prompts/templates
tokenizer_config.json: 100%|████████████████████████████████████████████████████| 321/321 [00:00<00:00, 1.93MB/s]
vocab.txt: 100%|███████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 652kB/s]
tokenizer.json: 100%|█████████████████████████████████████████████████████████| 466k/466k [00:00<00:00, 1.29MB/s]
special_tokens_map.json: 100%|███████████████████████████████████████████████████| 112/112 [00:00<00:00, 843kB/s]
config.json: 100%|██████████████████████████████████████████████████████████████| 619/619 [00:00<00:00, 4.83MB/s]
pytorch_model.bin: 100%|███████████████████████████████████████████████████████| 438M/438M [00:01<00:00, 282MB/s]
INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_facebook_contriever/chunk_embeddings
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_facebook_contriever/entity_embeddings
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_facebook_contriever/fact_embeddings
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/perrywu12/HippoRAG/src/hipporag/prompts/templates
INFO:src.hipporag.HippoRAG:Indexing Documents
INFO:src.hipporag.HippoRAG:Performing OpenIE
INFO:src.hipporag.embedding_store:Inserting 100 new records, 0 records already exist.
Batch Encoding: 104it [00:01, 70.45it/s]                                                                         
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 100 records to outputs/itsupport/gpt-4o-mini_facebook_contriever/chunk_embeddings/vdb_chunk.parquet
INFO:src.hipporag.HippoRAG:OpenIE results saved to outputs/itsupport/openie_results_ner_gpt-4o-mini.json
INFO:src.hipporag.HippoRAG:Encoding Entities
INFO:src.hipporag.embedding_store:Inserting 1298 new records, 0 records already exist.
Batch Encoding: 1304it [00:01, 861.87it/s]                                                                       
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1298 records to outputs/itsupport/gpt-4o-mini_facebook_contriever/entity_embeddings/vdb_entity.parquet
INFO:src.hipporag.HippoRAG:Encoding Facts
INFO:src.hipporag.embedding_store:Inserting 1456 new records, 0 records already exist.
Batch Encoding: 100%|███████████████████████████████████████████████████████| 1456/1456 [00:01<00:00, 855.16it/s]
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1456 records to outputs/itsupport/gpt-4o-mini_facebook_contriever/fact_embeddings/vdb_fact.parquet
INFO:src.hipporag.HippoRAG:Constructing Graph
INFO:src.hipporag.HippoRAG:Adding OpenIE triples to graph.
100it [00:00, 16420.56it/s]
INFO:src.hipporag.HippoRAG:Connecting passage nodes to phrase nodes.
100it [00:00, 43098.07it/s]
INFO:src.hipporag.HippoRAG:Found 100 new chunks to save into graph.
INFO:src.hipporag.HippoRAG:Expanding graph with synonymy edges
INFO:src.hipporag.HippoRAG:Performing KNN retrieval for each phrase nodes (1298).
KNN for Queries: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  4.48it/s]
100%|████████████████████████████████████████████████████████████████████| 1298/1298 [00:00<00:00, 190383.50it/s]
INFO:src.hipporag.HippoRAG:Graph construction completed!
{'num_phrase_nodes': 1298, 'num_passage_nodes': 100, 'num_total_nodes': 1398, 'num_extracted_triples': 1456, 'num_triples_with_passage_node': 1832, 'num_synonymy_triples': 2337, 'num_total_triples': 5625}
INFO:src.hipporag.HippoRAG:Writing graph with 1398 nodes, 5622 edges
INFO:src.hipporag.HippoRAG:Saving graph completed!
INFO:src.hipporag.HippoRAG:Preparing for fast retrieval.
INFO:src.hipporag.HippoRAG:Loading keys.
INFO:src.hipporag.HippoRAG:Loading embeddings.
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_fact.
Batch Encoding: 24it [00:00, 434.24it/s]                                                                         
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_passage.
Batch Encoding: 24it [00:00, 850.34it/s]                                                                         
Retrieving:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:   6%|███▊                                                             | 1/17 [00:01<00:21,  1.37s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  12%|███████▋                                                         | 2/17 [00:02<00:19,  1.29s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  18%|███████████▍                                                     | 3/17 [00:03<00:18,  1.31s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  24%|███████████████▎                                                 | 4/17 [00:05<00:16,  1.26s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  29%|███████████████████                                              | 5/17 [00:07<00:17,  1.50s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  35%|██████████████████████▉                                          | 6/17 [00:08<00:14,  1.36s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  41%|██████████████████████████▊                                      | 7/17 [00:10<00:15,  1.56s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  47%|██████████████████████████████▌                                  | 8/17 [00:11<00:13,  1.52s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:src.hipporag.HippoRAG:No facts found after reranking, return DPR results
Retrieving:  53%|██████████████████████████████████▍                              | 9/17 [00:12<00:10,  1.32s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  59%|█████████████████████████████████████▋                          | 10/17 [00:13<00:09,  1.30s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  65%|█████████████████████████████████████████▍                      | 11/17 [00:14<00:07,  1.28s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  71%|█████████████████████████████████████████████▏                  | 12/17 [00:16<00:06,  1.31s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:src.hipporag.HippoRAG:No facts found after reranking, return DPR results
Retrieving:  76%|████████████████████████████████████████████████▉               | 13/17 [00:17<00:04,  1.14s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  82%|████████████████████████████████████████████████████▋           | 14/17 [00:18<00:03,  1.15s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  88%|████████████████████████████████████████████████████████▍       | 15/17 [00:19<00:02,  1.18s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving:  94%|████████████████████████████████████████████████████████████▏   | 16/17 [00:20<00:01,  1.21s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:22<00:00,  1.30s/it]
INFO:src.hipporag.HippoRAG:Total Retrieval Time 22.15s
INFO:src.hipporag.HippoRAG:Total Recognition Memory Time 20.30s
INFO:src.hipporag.HippoRAG:Total PPR Time 1.70s
INFO:src.hipporag.HippoRAG:Total Misc Time 0.15s
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
Collecting QA prompts: 100%|███████████████████████████████████████████████████| 17/17 [00:00<00:00, 7433.61it/s]
QA Reading:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:   6%|███▊                                                             | 1/17 [00:01<00:24,  1.56s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  12%|███████▋                                                         | 2/17 [00:03<00:25,  1.68s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  18%|███████████▍                                                     | 3/17 [00:04<00:20,  1.44s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  24%|███████████████▎                                                 | 4/17 [00:06<00:19,  1.48s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  29%|███████████████████                                              | 5/17 [00:07<00:18,  1.57s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  35%|██████████████████████▉                                          | 6/17 [00:08<00:15,  1.38s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  41%|██████████████████████████▊                                      | 7/17 [00:10<00:15,  1.60s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  47%|██████████████████████████████▌                                  | 8/17 [00:12<00:14,  1.59s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  53%|██████████████████████████████████▍                              | 9/17 [00:13<00:11,  1.49s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  59%|█████████████████████████████████████▋                          | 10/17 [00:15<00:10,  1.51s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  65%|█████████████████████████████████████████▍                      | 11/17 [00:16<00:09,  1.52s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  71%|█████████████████████████████████████████████▏                  | 12/17 [00:17<00:07,  1.44s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  76%|████████████████████████████████████████████████▉               | 13/17 [00:19<00:05,  1.46s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  82%|████████████████████████████████████████████████████▋           | 14/17 [00:20<00:04,  1.45s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading:  94%|████████████████████████████████████████████████████████████▏   | 16/17 [00:22<00:01,  1.21s/it]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:24<00:00,  1.45s/it]
Extraction Answers from LLM Response: 17it [00:00, 270087.76it/s]
INFO:src.hipporag.HippoRAG:Evaluation results for QA: {'ExactMatch': 0.2941, 'F1': 0.6972}
```