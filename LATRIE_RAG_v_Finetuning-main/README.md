# **LATRIE_RAG_v_Finetuning**

## **Project Goal:**

The main idea is to evaluate two approaches: Retrieval Augmented Generation (RAG) and
Finetuning. The purpose is to discover the advantages and disadvantages of the two methods
using open (Quantized) Large Language Models (e.g. Mistral) in the medical field.

**Finetuning (Causal Language Modeling):** models must be finetuned/pretrained on clinical
guidelines that will be provided.

**RAG:** clinical guidelines can be stored in a vector database (see example) and the most
relevant chunks can be given to the LLM to generate factually correct responses.