from hipporag import HippoRAG
import torch

from src. hipporag.utils.config_utils import BaseConfig
torch.cuda.empty_cache()
# Disable cuDNN benchmark to reduce memory usage
torch.backends.cudnn.benchmark = False
# Prepare datasets and evaluation
docs = [
    "Oliver Badman is a politician.",
    "George Rankin is a politician.",
    "Thomas Marwick is a politician.",
    "Cinderella attended the royal ball.",
    "The prince used the lost glass slipper to search the kingdom.",
    "When the slipper fit perfectly, Cinderella was reunited with the prince.",
    "Erik Hort's birthplace is Montebello.",
    "Marina is bom in Minsk.",
    "Montebello is a part of Rockland County."
]

save_dir = 'outputs'# Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
llm_model_name = 'gpt-4o-mini'#'DeepSeek-V3-Base' #'gpt-4o-mini' # Any OpenAI model name
embedding_model_name = 'GritLM'#'nvidia/NV-Embed-v2'# Embedding model name (NV-Embed, GritLM or Contriever for now)

#Startup a HippoRAG instance
hipporag = HippoRAG(global_config=BaseConfig(embedding_batch_size=5,
                                             synonymy_edge_query_batch_size=10,
                                             synonymy_edge_key_batch_size=10,
                                             synonymy_edge_topk=5,
                                             ),
                    save_dir=save_dir, 
                    llm_model_name=llm_model_name,
                    embedding_model_name=embedding_model_name) 

#Run indexing
hipporag.index(docs=docs)

#Separate Retrieval & QA
queries = [
    "What is George Rankin's occupation?",
    "How did Cinderella reach her happy ending?",
    "What county is Erik Hort's birthplace a part of?"
]

retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=2)
qa_results = hipporag.rag_qa(retrieval_results)

#Combined Retrieval & QA
rag_results = hipporag.rag_qa(queries=queries)

#For Evaluation
answers = [
    ["Politician"],
    ["By going to the ball."],
    ["Rockland County"]
]

gold_docs = [
    ["George Rankin is a politician."],
    ["Cinderella attended the royal ball.",
    "The prince used the lost glass slipper to search the kingdom.",
    "When the slipper fit perfectly, Cinderella was reunited with the prince."],
    ["Erik Hort's birthplace is Montebello.",
    "Montebello is a part of Rockland County."]
]

rag_results = hipporag.rag_qa(queries=queries, 
                              gold_docs=gold_docs,
                              gold_answers=answers)
print(rag_results)