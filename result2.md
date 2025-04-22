
```
(hipporag) carolsong1110@instance-20250422-073211:~/HippoRAG$ python main.py --dataset itsupport --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name intfloat/e5-mistral-7b-instruct
```
INFO:src.hipporag.HippoRAG:Creating working directory: outputs/itsupport/gpt-4o-mini_intfloat_e5-mistral-7b-instruct
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
tokenizer_config.json: 100%|████████████████████████████████████████████████████| 981/981 [00:00<00:00, 6.82MB/s]
tokenizer.model: 100%|████████████████████████████████████████████████████████| 493k/493k [00:00<00:00, 66.3MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████| 1.80M/1.80M [00:00<00:00, 4.02MB/s]
added_tokens.json: 100%|███████████████████████████████████████████████████████| 42.0/42.0 [00:00<00:00, 353kB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████| 168/168 [00:00<00:00, 1.43MB/s]
config.json: 100%|██████████████████████████████████████████████████████████████| 629/629 [00:00<00:00, 4.83MB/s]
model.safetensors.index.json: 100%|██████████████████████████████████████████| 23.3k/23.3k [00:00<00:00, 100MB/s]
model-00001-of-00002.safetensors: 100%|██████████████████████████████████████| 9.94G/9.94G [00:27<00:00, 359MB/s]
model-00002-of-00002.safetensors: 100%|█████████████████████████████████████| 4.28G/4.28G [03:02<00:00, 23.4MB/s]
Downloading shards: 100%|█████████████████████████████████████████████████████████| 2/2 [03:30<00:00, 105.49s/it]
INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.20s/it]
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_intfloat_e5-mistral-7b-instruct/chunk_embeddings
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_intfloat_e5-mistral-7b-instruct/entity_embeddings
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_intfloat_e5-mistral-7b-instruct/fact_embeddings
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
INFO:src.hipporag.HippoRAG:Indexing Documents
INFO:src.hipporag.HippoRAG:Performing OpenIE
INFO:src.hipporag.embedding_store:Inserting 100 new records, 0 records already exist.
Batch Encoding:   0%|                                                                    | 0/100 [00:00<?, ?it/s]Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Batch Encoding: 104it [00:26,  3.92it/s]                                                                         
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 100 records to outputs/itsupport/gpt-4o-mini_intfloat_e5-mistral-7b-instruct/chunk_embeddings/vdb_chunk.parquet
NER:   0%|                                                                               | 0/100 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
NER:   1%| | 1/100 [00:01<02:38,  1.60s/it, total_prompt_tokens=575, total_completion_tokens=29, num_cache_hit=0]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
NER: 100%|█| 100/100 [00:08<00:00, 11.39it/s, total_prompt_tokens=69379, total_completion_tokens=4838, num_cache_
Extracting triples:   0%|                                                                | 0/100 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Extracting triples:   1%| | 1/100 [00:02<04:32,  2.75s/it, total_prompt_tokens=1011, total_completion_tokens=140,INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Extracting triples: 100%|█| 100/100 [00:25<00:00,  3.86it/s, total_prompt_tokens=102252, total_completion_tokens=
INFO:src.hipporag.HippoRAG:OpenIE results saved to outputs/itsupport/openie_results_ner_gpt-4o-mini.json
INFO:src.hipporag.HippoRAG:Encoding Entities
INFO:src.hipporag.embedding_store:Inserting 1315 new records, 0 records already exist.
Batch Encoding: 1320it [00:12, 109.42it/s]                                                                       
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1315 records to outputs/itsupport/gpt-4o-mini_intfloat_e5-mistral-7b-instruct/entity_embeddings/vdb_entity.parquet
INFO:src.hipporag.HippoRAG:Encoding Facts
INFO:src.hipporag.embedding_store:Inserting 1423 new records, 0 records already exist.
Batch Encoding: 1424it [00:16, 84.39it/s]                                                                        
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1423 records to outputs/itsupport/gpt-4o-mini_intfloat_e5-mistral-7b-instruct/fact_embeddings/vdb_fact.parquet
INFO:src.hipporag.HippoRAG:Constructing Graph
INFO:src.hipporag.HippoRAG:Adding OpenIE triples to graph.
100it [00:00, 16140.01it/s]
INFO:src.hipporag.HippoRAG:Connecting passage nodes to phrase nodes.
100it [00:00, 37684.67it/s]
INFO:src.hipporag.HippoRAG:Found 100 new chunks to save into graph.
INFO:src.hipporag.HippoRAG:Expanding graph with synonymy edges
INFO:src.hipporag.HippoRAG:Performing KNN retrieval for each phrase nodes (1315).
KNN for Queries: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  4.22it/s]
100%|█████████████████████████████████████████████████████████████████████| 1315/1315 [00:00<00:00, 42554.99it/s]
INFO:src.hipporag.HippoRAG:Graph construction completed!
{'num_phrase_nodes': 1315, 'num_passage_nodes': 100, 'num_total_nodes': 1415, 'num_extracted_triples': 1423, 'num_triples_with_passage_node': 1846, 'num_synonymy_triples': 8743, 'num_total_triples': 12012}
INFO:src.hipporag.HippoRAG:Writing graph with 1415 nodes, 12011 edges
INFO:src.hipporag.HippoRAG:Saving graph completed!
INFO:src.hipporag.HippoRAG:Preparing for fast retrieval.
INFO:src.hipporag.HippoRAG:Loading keys.
INFO:src.hipporag.HippoRAG:Loading embeddings.
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_fact.
Batch Encoding: 24it [00:00, 83.09it/s]                                                                          
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_passage.
Batch Encoding: 24it [00:00, 83.14it/s]                                                                          
Retrieving:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:21<00:00,  1.26s/it]
INFO:src.hipporag.HippoRAG:Total Retrieval Time 22.09s
INFO:src.hipporag.HippoRAG:Total Recognition Memory Time 21.19s
INFO:src.hipporag.HippoRAG:Total PPR Time 0.17s
INFO:src.hipporag.HippoRAG:Total Misc Time 0.73s
WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
INFO:src.hipporag.HippoRAG:Evaluation results for retrieval: {'Recall@1': 0.0, 'Recall@2': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0, 'Recall@20': 0.0, 'Recall@30': 0.0, 'Recall@50': 0.0, 'Recall@100': 0.0, 'Recall@150': 0.0, 'Recall@200': 0.0}
Collecting QA prompts: 100%|███████████████████████████████████████████████████| 17/17 [00:00<00:00, 6595.43it/s]
QA Reading:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:37<00:00,  2.22s/it]
Extraction Answers from LLM Response: 17it [00:00, 312733.19it/s]
**INFO:src.hipporag.HippoRAG:Evaluation results for QA: {'ExactMatch': 0.2353, 'F1': 0.5478}**

```
(hipporag) carolsong1110@instance-20250422-073211:~/HippoRAG$ python main.py --dataset itsupport --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name GritLM/GritLM-7B
```
INFO:src.hipporag.HippoRAG:Creating working directory: outputs/itsupport/gpt-4o-mini_GritLM_GritLM-7B
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
config.json: 100%|██████████████████████████████████████████████████████████████| 946/946 [00:00<00:00, 6.06MB/s]
modeling_gritlm7b.py: 100%|█████████████████████████████████████████████████| 65.2k/65.2k [00:00<00:00, 22.5MB/s]
A new version of the following files was downloaded from https://huggingface.co/GritLM/GritLM-7B:
- modeling_gritlm7b.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
model.safetensors.index.json: 100%|██████████████████████████████████████████| 23.9k/23.9k [00:00<00:00, 100MB/s]
model-00001-of-00003.safetensors: 100%|██████████████████████████████████████| 4.94G/4.94G [00:19<00:00, 253MB/s]
model-00002-of-00003.safetensors: 100%|██████████████████████████████████████| 5.00G/5.00G [00:17<00:00, 281MB/s]
model-00003-of-00003.safetensors: 100%|██████████████████████████████████████| 4.54G/4.54G [00:16<00:00, 274MB/s]
Downloading shards: 100%|██████████████████████████████████████████████████████████| 3/3 [00:54<00:00, 18.16s/it]
INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 3/3 [00:04<00:00,  1.57s/it]
generation_config.json: 100%|████████████████████████████████████████████████████| 111/111 [00:00<00:00, 843kB/s]
Created GritLM: torch.bfloat16 dtype, mean pool, unified mode, bbcc attn
tokenizer_config.json: 100%|████████████████████████████████████████████████| 1.35k/1.35k [00:00<00:00, 10.2MB/s]
tokenizer.model: 100%|████████████████████████████████████████████████████████| 493k/493k [00:00<00:00, 76.4MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████| 1.80M/1.80M [00:00<00:00, 3.02MB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████| 436/436 [00:00<00:00, 2.69MB/s]
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_GritLM_GritLM-7B/chunk_embeddings
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_GritLM_GritLM-7B/entity_embeddings
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_GritLM_GritLM-7B/fact_embeddings
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
INFO:src.hipporag.HippoRAG:Indexing Documents
INFO:src.hipporag.HippoRAG:Performing OpenIE
INFO:src.hipporag.embedding_store:Inserting 100 new records, 0 records already exist.
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 100 records to outputs/itsupport/gpt-4o-mini_GritLM_GritLM-7B/chunk_embeddings/vdb_chunk.parquet
INFO:src.hipporag.HippoRAG:OpenIE results saved to outputs/itsupport/openie_results_ner_gpt-4o-mini.json
INFO:src.hipporag.HippoRAG:Encoding Entities
INFO:src.hipporag.embedding_store:Inserting 1315 new records, 0 records already exist.
Batches: 100%|█████████████████████████████████████████████████████████████████| 165/165 [00:11<00:00, 13.83it/s]
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1315 records to outputs/itsupport/gpt-4o-mini_GritLM_GritLM-7B/entity_embeddings/vdb_entity.parquet
INFO:src.hipporag.HippoRAG:Encoding Facts
INFO:src.hipporag.embedding_store:Inserting 1423 new records, 0 records already exist.
Batches: 100%|█████████████████████████████████████████████████████████████████| 178/178 [00:15<00:00, 11.29it/s]
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1423 records to outputs/itsupport/gpt-4o-mini_GritLM_GritLM-7B/fact_embeddings/vdb_fact.parquet
INFO:src.hipporag.HippoRAG:Constructing Graph
INFO:src.hipporag.HippoRAG:Adding OpenIE triples to graph.
100it [00:00, 17447.91it/s]
INFO:src.hipporag.HippoRAG:Connecting passage nodes to phrase nodes.
100it [00:00, 46228.41it/s]
INFO:src.hipporag.HippoRAG:Found 100 new chunks to save into graph.
INFO:src.hipporag.HippoRAG:Expanding graph with synonymy edges
INFO:src.hipporag.HippoRAG:Performing KNN retrieval for each phrase nodes (1315).
KNN for Queries: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.77it/s]
100%|█████████████████████████████████████████████████████████████████████| 1315/1315 [00:00<00:00, 54213.94it/s]
INFO:src.hipporag.HippoRAG:Graph construction completed!
{'num_phrase_nodes': 1315, 'num_passage_nodes': 100, 'num_total_nodes': 1415, 'num_extracted_triples': 1423, 'num_triples_with_passage_node': 1846, 'num_synonymy_triples': 8934, 'num_total_triples': 12203}
INFO:src.hipporag.HippoRAG:Writing graph with 1415 nodes, 12202 edges
INFO:src.hipporag.HippoRAG:Saving graph completed!
INFO:src.hipporag.HippoRAG:Preparing for fast retrieval.
INFO:src.hipporag.HippoRAG:Loading keys.
INFO:src.hipporag.HippoRAG:Loading embeddings.
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_fact.
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_passage.
Retrieving:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:21<00:00,  1.27s/it]
INFO:src.hipporag.HippoRAG:Total Retrieval Time 22.38s
INFO:src.hipporag.HippoRAG:Total Recognition Memory Time 20.87s
INFO:src.hipporag.HippoRAG:Total PPR Time 0.61s
INFO:src.hipporag.HippoRAG:Total Misc Time 0.90s
WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
INFO:src.hipporag.HippoRAG:Evaluation results for retrieval: {'Recall@1': 0.0, 'Recall@2': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0, 'Recall@20': 0.0, 'Recall@30': 0.0, 'Recall@50': 0.0, 'Recall@100': 0.0, 'Recall@150': 0.0, 'Recall@200': 0.0}
Collecting QA prompts: 100%|███████████████████████████████████████████████████| 17/17 [00:00<00:00, 6233.34it/s]
QA Reading:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:27<00:00,  1.59s/it]
Extraction Answers from LLM Response: 17it [00:00, 331642.64it/s]
**INFO:src.hipporag.HippoRAG:Evaluation results for QA: {'ExactMatch': 0.3529, 'F1': 0.6959}**

```
(hipporag) carolsong1110@instance-20250422-073211:~/HippoRAG$ python main.py --dataset itsupport --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name BAAI/bge-small-en-v1.5
```
INFO:src.hipporag.HippoRAG:Creating working directory: outputs/itsupport/gpt-4o-mini_BAAI_bge-small-en-v1.5
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
tokenizer_config.json: 100%|████████████████████████████████████████████████████| 366/366 [00:00<00:00, 2.41MB/s]
vocab.txt: 100%|███████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 780kB/s]
tokenizer.json: 100%|█████████████████████████████████████████████████████████| 711k/711k [00:00<00:00, 4.62MB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████| 125/125 [00:00<00:00, 1.04MB/s]
config.json: 100%|██████████████████████████████████████████████████████████████| 743/743 [00:00<00:00, 5.84MB/s]
model.safetensors: 100%|███████████████████████████████████████████████████████| 133M/133M [00:00<00:00, 393MB/s]
INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_BAAI_bge-small-en-v1.5/chunk_embeddings
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_BAAI_bge-small-en-v1.5/entity_embeddings
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_BAAI_bge-small-en-v1.5/fact_embeddings
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
INFO:src.hipporag.HippoRAG:Indexing Documents
INFO:src.hipporag.HippoRAG:Performing OpenIE
INFO:src.hipporag.embedding_store:Inserting 100 new records, 0 records already exist.
Batch Encoding: 104it [00:00, 157.47it/s]                                                                        
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 100 records to outputs/itsupport/gpt-4o-mini_BAAI_bge-small-en-v1.5/chunk_embeddings/vdb_chunk.parquet
INFO:src.hipporag.HippoRAG:OpenIE results saved to outputs/itsupport/openie_results_ner_gpt-4o-mini.json
INFO:src.hipporag.HippoRAG:Encoding Entities
INFO:src.hipporag.embedding_store:Inserting 1315 new records, 0 records already exist.
Batch Encoding: 1320it [00:01, 894.34it/s]                                                                       
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1315 records to outputs/itsupport/gpt-4o-mini_BAAI_bge-small-en-v1.5/entity_embeddings/vdb_entity.parquet
INFO:src.hipporag.HippoRAG:Encoding Facts
INFO:src.hipporag.embedding_store:Inserting 1423 new records, 0 records already exist.
Batch Encoding: 1424it [00:01, 861.45it/s]                                                                       
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1423 records to outputs/itsupport/gpt-4o-mini_BAAI_bge-small-en-v1.5/fact_embeddings/vdb_fact.parquet
INFO:src.hipporag.HippoRAG:Constructing Graph
INFO:src.hipporag.HippoRAG:Adding OpenIE triples to graph.
100it [00:00, 17137.09it/s]
INFO:src.hipporag.HippoRAG:Connecting passage nodes to phrase nodes.
100it [00:00, 44276.41it/s]
INFO:src.hipporag.HippoRAG:Found 100 new chunks to save into graph.
INFO:src.hipporag.HippoRAG:Expanding graph with synonymy edges
INFO:src.hipporag.HippoRAG:Performing KNN retrieval for each phrase nodes (1315).
KNN for Queries: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  4.57it/s]
100%|█████████████████████████████████████████████████████████████████████| 1315/1315 [00:00<00:00, 79528.06it/s]
INFO:src.hipporag.HippoRAG:Graph construction completed!
{'num_phrase_nodes': 1315, 'num_passage_nodes': 100, 'num_total_nodes': 1415, 'num_extracted_triples': 1423, 'num_triples_with_passage_node': 1846, 'num_synonymy_triples': 4359, 'num_total_triples': 7628}
INFO:src.hipporag.HippoRAG:Writing graph with 1415 nodes, 7627 edges
INFO:src.hipporag.HippoRAG:Saving graph completed!
INFO:src.hipporag.HippoRAG:Preparing for fast retrieval.
INFO:src.hipporag.HippoRAG:Loading keys.
INFO:src.hipporag.HippoRAG:Loading embeddings.
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_fact.
Batch Encoding: 24it [00:00, 773.98it/s]                                                                         
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_passage.
Batch Encoding: 24it [00:00, 901.22it/s]                                                                         
Retrieving:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:26<00:00,  1.56s/it]
INFO:src.hipporag.HippoRAG:Total Retrieval Time 26.58s
INFO:src.hipporag.HippoRAG:Total Recognition Memory Time 26.40s
INFO:src.hipporag.HippoRAG:Total PPR Time 0.05s
INFO:src.hipporag.HippoRAG:Total Misc Time 0.13s
WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
INFO:src.hipporag.HippoRAG:Evaluation results for retrieval: {'Recall@1': 0.0, 'Recall@2': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0, 'Recall@20': 0.0, 'Recall@30': 0.0, 'Recall@50': 0.0, 'Recall@100': 0.0, 'Recall@150': 0.0, 'Recall@200': 0.0}
Collecting QA prompts: 100%|███████████████████████████████████████████████████| 17/17 [00:00<00:00, 6064.23it/s]
QA Reading:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:27<00:00,  1.61s/it]
Extraction Answers from LLM Response: 17it [00:00, 250186.55it/s]
**INFO:src.hipporag.HippoRAG:Evaluation results for QA: {'ExactMatch': 0.3529, 'F1': 0.6919}**

```
(hipporag) carolsong1110@instance-20250422-073211:~/HippoRAG$ python main.py --dataset itsupport
INFO:src.hipporag.HippoRAG:Creating working directory: outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2
```
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
config.json: 100%|██████████████████████████████████████████████████████████| 2.66k/2.66k [00:00<00:00, 16.8MB/s]
configuration_nvembed.py: 100%|█████████████████████████████████████████████| 3.20k/3.20k [00:00<00:00, 27.0MB/s]
A new version of the following files was downloaded from https://huggingface.co/nvidia/NV-Embed-v2:
- configuration_nvembed.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
modeling_nvembed.py: 100%|██████████████████████████████████████████████████| 18.7k/18.7k [00:00<00:00, 9.20MB/s]
INFO:datasets:PyTorch version 2.5.1 available.
INFO:datasets:Polars version 1.27.1 available.
A new version of the following files was downloaded from https://huggingface.co/nvidia/NV-Embed-v2:
- modeling_nvembed.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
model.safetensors.index.json: 100%|█████████████████████████████████████████| 28.2k/28.2k [00:00<00:00, 4.54MB/s]
model-00001-of-00004.safetensors: 100%|█████████████████████████████████████| 5.00G/5.00G [03:50<00:00, 21.6MB/s]
model-00002-of-00004.safetensors: 100%|█████████████████████████████████████| 4.92G/4.92G [03:09<00:00, 26.0MB/s]
model-00003-of-00004.safetensors: 100%|█████████████████████████████████████| 5.00G/5.00G [02:37<00:00, 31.7MB/s]
model-00004-of-00004.safetensors: 100%|████████████████████████████████████████| 789M/789M [00:02<00:00, 376MB/s]
Downloading shards: 100%|█████████████████████████████████████████████████████████| 4/4 [09:41<00:00, 145.50s/it]
tokenizer_config.json: 100%|████████████████████████████████████████████████████| 997/997 [00:00<00:00, 7.69MB/s]
tokenizer.model: 100%|████████████████████████████████████████████████████████| 493k/493k [00:00<00:00, 53.7MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████| 1.80M/1.80M [00:00<00:00, 4.13MB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████| 551/551 [00:00<00:00, 4.04MB/s]
INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 4/4 [00:05<00:00,  1.31s/it]
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/chunk_embeddings
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/entity_embeddings
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/fact_embeddings
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
INFO:src.hipporag.HippoRAG:Indexing Documents
INFO:src.hipporag.HippoRAG:Performing OpenIE
INFO:src.hipporag.embedding_store:Inserting 100 new records, 0 records already exist.
Batch Encoding:   0%|                                                                    | 0/100 [00:00<?, ?it/s]/home/carolsong1110/.cache/huggingface/modules/transformers_modules/nvidia/NV-Embed-v2/c50d55f43bde7e6a18e0eaa15a62fd63a930f1a1/modeling_nvembed.py:349: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  'input_ids': torch.tensor(batch_dict.get('input_ids').to(batch_dict.get('input_ids')).long()),
/home/carolsong1110/miniconda3/envs/hipporag/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
Batch Encoding: 104it [00:29,  3.55it/s]                                                                         
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 100 records to outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet
INFO:src.hipporag.HippoRAG:OpenIE results saved to outputs/itsupport/openie_results_ner_gpt-4o-mini.json
INFO:src.hipporag.HippoRAG:Encoding Entities
INFO:src.hipporag.embedding_store:Inserting 1315 new records, 0 records already exist.
Batch Encoding: 1320it [00:21, 61.93it/s]                                                                        
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1315 records to outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/entity_embeddings/vdb_entity.parquet
INFO:src.hipporag.HippoRAG:Encoding Facts
INFO:src.hipporag.embedding_store:Inserting 1423 new records, 0 records already exist.
Batch Encoding: 1424it [00:28, 49.29it/s]                                                                        
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1423 records to outputs/itsupport/gpt-4o-mini_nvidia_NV-Embed-v2/fact_embeddings/vdb_fact.parquet
INFO:src.hipporag.HippoRAG:Constructing Graph
INFO:src.hipporag.HippoRAG:Adding OpenIE triples to graph.
100it [00:00, 16984.43it/s]
INFO:src.hipporag.HippoRAG:Connecting passage nodes to phrase nodes.
100it [00:00, 44257.72it/s]
INFO:src.hipporag.HippoRAG:Found 100 new chunks to save into graph.
INFO:src.hipporag.HippoRAG:Expanding graph with synonymy edges
INFO:src.hipporag.HippoRAG:Performing KNN retrieval for each phrase nodes (1315).
KNN for Queries: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.25it/s]
100%|█████████████████████████████████████████████████████████████████████| 1315/1315 [00:00<00:00, 62247.59it/s]
INFO:src.hipporag.HippoRAG:Graph construction completed!
{'num_phrase_nodes': 1315, 'num_passage_nodes': 100, 'num_total_nodes': 1415, 'num_extracted_triples': 1423, 'num_triples_with_passage_node': 1846, 'num_synonymy_triples': 5957, 'num_total_triples': 9226}
INFO:src.hipporag.HippoRAG:Writing graph with 1415 nodes, 9225 edges
INFO:src.hipporag.HippoRAG:Saving graph completed!
INFO:src.hipporag.HippoRAG:Preparing for fast retrieval.
INFO:src.hipporag.HippoRAG:Loading keys.
INFO:src.hipporag.HippoRAG:Loading embeddings.
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_fact.
Batch Encoding: 24it [00:00, 47.82it/s]                                                                          
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_passage.
Batch Encoding: 24it [00:00, 45.87it/s]                                                                          
Retrieving:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:26<00:00,  1.54s/it]
INFO:src.hipporag.HippoRAG:Total Retrieval Time 27.40s
INFO:src.hipporag.HippoRAG:Total Recognition Memory Time 25.98s
INFO:src.hipporag.HippoRAG:Total PPR Time 0.18s
INFO:src.hipporag.HippoRAG:Total Misc Time 1.24s
WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
INFO:src.hipporag.HippoRAG:Evaluation results for retrieval: {'Recall@1': 0.0, 'Recall@2': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0, 'Recall@20': 0.0, 'Recall@30': 0.0, 'Recall@50': 0.0, 'Recall@100': 0.0, 'Recall@150': 0.0, 'Recall@200': 0.0}
Collecting QA prompts: 100%|███████████████████████████████████████████████████| 17/17 [00:00<00:00, 2587.39it/s]
QA Reading:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:30<00:00,  1.79s/it]
Extraction Answers from LLM Response: 17it [00:00, 318317.71it/s]
**INFO:src.hipporag.HippoRAG:Evaluation results for QA: {'ExactMatch': 0.3529, 'F1': 0.6823}**

```
(hipporag) carolsong1110@instance-20250422-073211:~/HippoRAG$ python main.py --dataset itsupport --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name Facebook/contriever
```
INFO:src.hipporag.HippoRAG:Creating working directory: outputs/itsupport/gpt-4o-mini_Facebook_contriever
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
tokenizer_config.json: 100%|████████████████████████████████████████████████████| 321/321 [00:00<00:00, 2.22MB/s]
vocab.txt: 100%|███████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 761kB/s]
tokenizer.json: 100%|█████████████████████████████████████████████████████████| 466k/466k [00:00<00:00, 8.78MB/s]
special_tokens_map.json: 100%|███████████████████████████████████████████████████| 112/112 [00:00<00:00, 881kB/s]
config.json: 100%|██████████████████████████████████████████████████████████████| 619/619 [00:00<00:00, 4.81MB/s]
pytorch_model.bin: 100%|███████████████████████████████████████████████████████| 438M/438M [00:01<00:00, 402MB/s]
INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_Facebook_contriever/chunk_embeddings
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_Facebook_contriever/entity_embeddings
INFO:src.hipporag.embedding_store:Creating working directory: outputs/itsupport/gpt-4o-mini_Facebook_contriever/fact_embeddings
INFO:src.hipporag.prompts.prompt_template_manager:Loading templates from directory: /home/carolsong1110/HippoRAG/src/hipporag/prompts/templates
INFO:src.hipporag.HippoRAG:Indexing Documents
INFO:src.hipporag.HippoRAG:Performing OpenIE
INFO:src.hipporag.embedding_store:Inserting 100 new records, 0 records already exist.
Batch Encoding: 104it [00:01, 71.65it/s]                                                                         
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 100 records to outputs/itsupport/gpt-4o-mini_Facebook_contriever/chunk_embeddings/vdb_chunk.parquet
INFO:src.hipporag.HippoRAG:OpenIE results saved to outputs/itsupport/openie_results_ner_gpt-4o-mini.json
INFO:src.hipporag.HippoRAG:Encoding Entities
INFO:src.hipporag.embedding_store:Inserting 1315 new records, 0 records already exist.
Batch Encoding: 1320it [00:01, 885.63it/s]                                                                       
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1315 records to outputs/itsupport/gpt-4o-mini_Facebook_contriever/entity_embeddings/vdb_entity.parquet
INFO:src.hipporag.HippoRAG:Encoding Facts
INFO:src.hipporag.embedding_store:Inserting 1423 new records, 0 records already exist.
Batch Encoding: 1424it [00:01, 851.90it/s]                                                                       
INFO:src.hipporag.embedding_store:Saving new records.
INFO:src.hipporag.embedding_store:Saved 1423 records to outputs/itsupport/gpt-4o-mini_Facebook_contriever/fact_embeddings/vdb_fact.parquet
INFO:src.hipporag.HippoRAG:Constructing Graph
INFO:src.hipporag.HippoRAG:Adding OpenIE triples to graph.
100it [00:00, 16507.81it/s]
INFO:src.hipporag.HippoRAG:Connecting passage nodes to phrase nodes.
100it [00:00, 43253.62it/s]
INFO:src.hipporag.HippoRAG:Found 100 new chunks to save into graph.
INFO:src.hipporag.HippoRAG:Expanding graph with synonymy edges
INFO:src.hipporag.HippoRAG:Performing KNN retrieval for each phrase nodes (1315).
KNN for Queries: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  4.55it/s]
100%|████████████████████████████████████████████████████████████████████| 1315/1315 [00:00<00:00, 198671.20it/s]
INFO:src.hipporag.HippoRAG:Graph construction completed!
{'num_phrase_nodes': 1315, 'num_passage_nodes': 100, 'num_total_nodes': 1415, 'num_extracted_triples': 1423, 'num_triples_with_passage_node': 1846, 'num_synonymy_triples': 2294, 'num_total_triples': 5563}
INFO:src.hipporag.HippoRAG:Writing graph with 1415 nodes, 5562 edges
INFO:src.hipporag.HippoRAG:Saving graph completed!
INFO:src.hipporag.HippoRAG:Preparing for fast retrieval.
INFO:src.hipporag.HippoRAG:Loading keys.
INFO:src.hipporag.HippoRAG:Loading embeddings.
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_fact.
Batch Encoding: 24it [00:00, 725.60it/s]                                                                         
INFO:src.hipporag.HippoRAG:Encoding 17 queries for query_to_passage.
Batch Encoding: 24it [00:00, 882.35it/s]                                                                         
Retrieving:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Retrieving: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:29<00:00,  1.74s/it]
INFO:src.hipporag.HippoRAG:Total Retrieval Time 29.66s
INFO:src.hipporag.HippoRAG:Total Recognition Memory Time 28.89s
INFO:src.hipporag.HippoRAG:Total PPR Time 0.63s
INFO:src.hipporag.HippoRAG:Total Misc Time 0.14s
WARNING:src.hipporag.evaluation.retrieval_eval:Length of retrieved docs (100) is smaller than largest topk for recall score (200)
INFO:src.hipporag.HippoRAG:Evaluation results for retrieval: {'Recall@1': 0.0, 'Recall@2': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0, 'Recall@20': 0.0, 'Recall@30': 0.0, 'Recall@50': 0.0, 'Recall@100': 0.0, 'Recall@150': 0.0, 'Recall@200': 0.0}
Collecting QA prompts: 100%|███████████████████████████████████████████████████| 17/17 [00:00<00:00, 7344.03it/s]
QA Reading:   0%|                                                                         | 0/17 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
QA Reading: 100%|████████████████████████████████████████████████████████████████| 17/17 [00:30<00:00,  1.78s/it]
Extraction Answers from LLM Response: 17it [00:00, 325585.24it/s]
**INFO:src.hipporag.HippoRAG:Evaluation results for QA: {'ExactMatch': 0.2353, 'F1': 0.6588}**