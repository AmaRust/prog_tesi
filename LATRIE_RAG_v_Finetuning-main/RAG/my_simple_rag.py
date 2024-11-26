from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import torch
from llama_index.core.vector_stores import VectorStoreQuery
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from llama_index.core import Document
from datasets import load_dataset

# hide the loading messages
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

# METRICS
import nltk
nltk.download('wordnet')
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score

import json
import os
from bs4 import BeautifulSoup
    
class HTMLDirectoryReader:
    def __init__(self, directory):
        self.directory = directory

    def load_data(self):
        documents = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".html") or filename.endswith(".htm"):
                file_path = os.path.join(self.directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, 'lxml')
                    self.remove_references(soup) 
                    text_content = self.extract_text_from_html(soup)
                    if text_content:  
                        documents.append(text_content)
        return documents

    def remove_references(self, soup):
        references_section = soup.find('h2', id='references')
        if references_section:
            next_sibling = references_section.find_next_sibling()
            references_section.decompose()
            while next_sibling :
                next_sibling_to_decompose = next_sibling
                next_sibling = next_sibling.find_next_sibling()
                next_sibling_to_decompose.decompose()

    def extract_text_from_html(self, soup):
        text = soup.get_text()
        
        return text

# query2doc method for query expansion    
def query2doc(query, k, llm, tokenizer):
    dataset = load_dataset("intfloat/query2doc_msmarco")
    train_dataset = dataset['train']
    sampled_train_dataset = train_dataset.shuffle(seed=42).select(range(k))
    
    passages = "\n".join([f"Query: {sampled_train_dataset[i]['query']}\nPassage: {sampled_train_dataset[i]['pseudo_doc']}\n" for i in range(k)])
    
    prompt = f'''[INST] 
Write a passage that answers the given query (at the end):\n
{passages}
Query: {query}
Passage:  
[/INST]'''
    
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = llm.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=500)

    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response = generated_text[len(prompt):].strip()
    
    new_query = f"{query} [SEP] {response}"

    return new_query


# chain of thought method 
def my_chain_of_thought(query, k, llm, tokenizer):
    dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")
    train_dataset = dataset['train']
    sampled_train_dataset = train_dataset.shuffle(seed=42).select(range(k))

    few_shot = "\n".join([f"Query: {sampled_train_dataset[i]['input']}\nAnswer: {sampled_train_dataset[i]['output']}\n" for i in range(k)])
    
    prompt = f'''[INST] 
For each answer, let's think step by step to complete them (in the answer area, replace the current text by your thinking):\n
{few_shot} 
[/INST]'''

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = llm.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=1000)

    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response = generated_text[len(prompt):].strip()

    new_prompt = f'''[INST] 
{response}\n
---\n
Query: {query}
Answer: Let's think step by step to answer.
[/INST]'''
    
    inputs = tokenizer(new_prompt, return_tensors="pt")

    outputs = llm.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=1000)

    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response = generated_text[len(new_prompt):].strip()

    return response

# reranking method to enhance the relevance of retrieved docs
def my_reranking(model_name, query, passages, top_n): # model_name : "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    features = tokenizer([query] * len(passages), passages,  padding=True, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        scores = model(**features).logits

    scores = scores.squeeze().tolist()

    scored_passages = list(zip(passages, scores))

    sorted_passages = sorted(scored_passages, key=lambda x: x[1], reverse=True)

    best_passages = []
    scores = []
    for passage, score in sorted_passages[:top_n]:
        best_passages.append(passage)
        scores.append(score)

    return (best_passages, scores)

    
def my_score_BLEU(reference, candidate):
    res = f"Individual 1-gram : {sentence_bleu(reference.split(), candidate, weights=(1, 0, 0, 0))}\nIndividual 2-gram : {sentence_bleu(reference.split(), candidate, weights=(0, 1, 0, 0))}\nIndividual 3-gram : {sentence_bleu(reference.split(), candidate, weights=(0, 0, 1, 0))}\nIndividual 4-gram : {sentence_bleu(reference.split(), candidate, weights=(0, 0, 0, 1))}"
    return res

def my_score_ROUGE(reference, candidate, scorer):
    scores = scorer.score(reference, candidate)
    res = f"ROUGE_1 : {scores['rouge1']}\nROUGE_L : {scores['rougeL']}"
    return res

def my_score_METEOR(reference, candidate):
    return meteor_score([reference.split()], candidate.split())

def my_BERTScore(reference, candidate):
    P, R, F1 = score([candidate], [reference], lang='en', rescale_with_baseline=True)
    res = f"Precision : {P.mean():.3f}\nRecall : {R.mean():.3f}\nF1 : {F1.mean():.3f}"
    return res 


def get_context_by_question(question, dataset):
    for entry in dataset:
        if entry["question"] == question:
            return entry["context"]
    return "Question not found."

def get_answer_by_question(question, dataset):
    for entry in dataset:
        if entry["question"] == question:
            return entry["answer"]
    return "Question not found."

if __name__ == "__main__":

    query = "How is the evidence on treatment of infections caused by 3GCephRE organized?"

    # load the JSON file
    with open('../data/assessment_set/dataset.json', 'r') as dataset:
        qa_pairs = json.load(dataset)
    set_size = len(qa_pairs)

    #--------------------------------------------
    # Beginning of RAG

    # load model
    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="main")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # #--------------------------------------------

    documents = HTMLDirectoryReader("../data/Medical_Guidelines/HTML").load_data()
        
    # Transforming text strings into Document objects
    i = 1
    document_objects = [Document(text=doc, title=f"Document {i+1}") for i, doc in enumerate(documents)]

    reference_context = get_context_by_question(query, qa_pairs)

    #--------------------------------------------

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    text_parser = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=25,
        # separator=" ",
    )

    text_chunks = []
    # maintain relationship with source doc index, to help inject doc metadata 
    doc_idxs = []
    for doc_idx, doc in enumerate(document_objects):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = document_objects[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    # create a faiss index
    d = len(nodes[0].embedding)  # dimension
    faiss_index = faiss.IndexFlatL2(d)

    vector_store = FaissVectorStore(faiss_index=faiss_index)

    vector_store.add(nodes)


    query_embedding = embed_model.get_query_embedding(query)

    query_mode = "default"
    # query_mode = "sparse"
    # query_mode = "hybrid"

    top_k = 5
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=top_k, mode=query_mode
    )

    # returns a VectorStoreQueryResult
    query_result = vector_store.query(vector_store_query)

    context = "Context:\n"

    chunks = []
    for i in range(top_k):
        chunk_text = nodes[int(query_result.ids[top_k-i-1])].text
        chunks.append(chunk_text)   # useful if reranking
        context += f"{chunk_text}\n\n"
        print(f"- CHUNK {i + 1} (similarity : {query_result.similarities[top_k-i-1]}):\n{chunk_text}\n\n")

    # With re-ranking
    print("AFTER RE-RANKING\n\n")

    top_n = 2
    context = "Context:\n"
    (new_chunks, scores) = my_reranking("cross-encoder/ms-marco-MiniLM-L-6-v2", query, chunks, top_n)

    for i in range(top_n):
        context += f"{new_chunks[i]}\n\n"
        print(f"- CHUNK {i + 1} (score: {scores[i]}) :\n{new_chunks[i]}\n\n")

    #--------------------------------------------
    model.eval()
    max_new_tokens = 190

    # prompt (no additional prompt and no context)
    
    prompt=f'''[INST] {query} [/INST]'''

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=max_new_tokens)

    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response1 = generated_text[len(prompt):].strip()

    print(f"* The answer obtained without additional prompt and context is :\n\n{response1}\n\n")

    # prompt (additional prompt and no context)

    intstructions_string = f"""You are an expert medical professional with extensive knowledge in various fields of medicine, \n
        You are here to provide accurate, evidence-based information and guidance on a wide range of medical topics. \n
        Preferably give clear answers without too many details.

        Please respond to the following question.
        """
    prompt_template = lambda question: f'''[INST] {intstructions_string} \n{question} \n[/INST]'''

    prompt = prompt_template(query)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=max_new_tokens)

    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response2 = generated_text[len(prompt):].strip()

    print(f"* The answer obtained with additional prompt but without context is :\n\n{response2}\n\n")

    # prompt (no additional prompt and context)

    prompt_template_w_context = lambda context, question: f"""[INST]
    {context}
    Please respond to the following question. Use the context above if it is helpful.

    {question}
    [/INST]
    """

    prompt = prompt_template_w_context(context, query)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=max_new_tokens)

    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response3 = generated_text[len(prompt):].strip()

    print(f"* The answer obtained with context but without additional prompt is :\n\n{response3}\n\n")

    # prompt (context and additional prompt)

    prompt_template_w_context = lambda context, question: f"""[INST]You are an expert medical professional with extensive knowledge in various fields of medicine, \n
    You are here to provide accurate, evidence-based information and guidance on a wide range of medical topics. \n
    Preferably give clear answers without too many details.

    {context}
    Please respond to the following question. Use the context above if it is helpful.

    {question}
    [/INST]
    """
        
    prompt = prompt_template_w_context(context, query)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=max_new_tokens)

    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response4 = generated_text[len(prompt):].strip()

    print(f"* The answer obtained with context and additional prompt is :\n\n{response4}\n\n")

    #--------------------------------------------

    # Use of metrics 
    reference_response = get_answer_by_question(query, qa_pairs)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    print(f"* NO ADDITIONAL PROMPT / NO CONTEXT\n")
    print(f"         - BLEU :")
    print(f"{my_score_BLEU(reference_response, response1)}\n")

    print(f"         - ROUGE :")
    print(f"{my_score_ROUGE(reference_response, response1, scorer)}\n")

    print(f"         - METEOR :" )
    print(f"{my_score_METEOR(reference_response, response1)}\n")

    print(f"         - BERTScore :")
    print(f"{my_BERTScore(reference_response, response1)}\n")

    print(f"* ADDITIONAL PROMPT / NO CONTEXT\n")
    print(f"         - BLEU :")
    print(f"{my_score_BLEU(reference_response, response2)}\n")

    print(f"         - ROUGE :")
    print(f"{my_score_ROUGE(reference_response, response2, scorer)}\n")

    print(f"         - METEOR :")
    print(f"{my_score_METEOR(reference_response, response2)}\n")

    print(f"         - BERTScore :")
    print(f"{my_BERTScore(reference_response, response2)}\n")

    print(f"* NO ADDITIONAL PROMPT / CONTEXT\n")
    print(f"         - BLEU :")
    print(f"{my_score_BLEU(reference_response, response3)}\n")

    print(f"         - ROUGE :")
    print(f"{my_score_ROUGE(reference_response, response3, scorer)}\n")

    print(f"         - METEOR :")
    print(f"{my_score_METEOR(reference_response, response3)}\n")

    print(f"         - BERTScore :")
    print(f"{my_BERTScore(reference_response, response3)}\n")

    print(f"* ADDITIONAL PROMPT / CONTEXT\n")
    print(f"         - BLEU :")
    print(f"{my_score_BLEU(reference_response, response4)}\n")

    print(f"         - ROUGE :")
    print(f"{my_score_ROUGE(reference_response, response4, scorer)}\n")

    print(f"         - METEOR :")
    print(f"{my_score_METEOR(reference_response, response4)}\n")

    print(f"         - BERTScore :")
    print(f"{my_BERTScore(reference_response, response4)}\n")

    