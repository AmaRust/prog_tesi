import sys
sys.path.insert(0, '../')
from utils import get_choice, HTMLDirectoryReader, get_answer_by_question, get_context_by_question, my_reranking, my_reranking_2, get_config_hash, save_embeddings, load_embeddings, Citation, QuotedAnswer, is_unique_list_of_ints, mean_max_tokens, get_max_new_tokens,compute_DCG, compute_NDCG, compute_MRR, compute_MAP, split_into_sentences
import configparser
import itertools
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
import multiprocessing


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import torch
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.node_parser import SemanticSplitterNodeParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline
from llama_index.core import Document
from datetime import datetime
from huggingface_hub import login
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

import numpy as np
import random
import json
import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

import glob
from collections import OrderedDict
import ast

import ollama
import re
from sklearn.metrics.pairwise import cosine_similarity

import bm25s
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize


def splitting_and_context(splitting_method, embedding_model_name, chunk_size, chunk_overlap, document_objects, top_k, query, reference_answer, qa_pairs, reranking, top_n, llm, model, tokenizer, device, retriever_method, retriever_evaluation):
    new_context = ""
    new_chunks = []
    new_ids = []
    retriever_rates = []
    if splitting_method == "Settings":
        ids = [0 for i in range(top_k)]
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)

        Settings.llm = None
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

        # store docs into vector DB
        index = VectorStoreIndex.from_documents(document_objects)

        # configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k, # number of docs to retreive
        )

        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
        ) 

        if isinstance(query, list):
            contexts = []
            new_contexts = []

            for i in query:
                response = query_engine.query(qa_pairs[i-1]['question'])
                context = "Context:\n"
                chunks = []
                for j in range(top_k):
                    chunk_text = response.source_nodes[j].text
                    chunks.append(chunk_text)
                    context += f"{chunk_text}\n\n"
                contexts.append(context)
                if retriever_evaluation:
                    _, retriever_rate = evaluate_retriever(qa_pairs[i-1]['question'], qa_pairs[i-1]['answer'], chunks, [], llm, model, tokenizer, device, get_max_new_tokens(llm, qa_pairs[i-1]['answer']))
                    retriever_rates.append(retriever_rate)

            if reranking > 0:
                for i in query:
                    response = query_engine.query(qa_pairs[i-1]['question'])
                    new_context = "Context:\n"
                    
                    new_chunks, scores = my_reranking("BAAI/bge-reranker-base", qa_pairs[i-1]['question'], chunks, top_n, []) if reranking == 1 else my_reranking_2(llm, model, tokenizer, device, qa_pairs[i-1]['question'], qa_pairs[i-1]['answer'], chunks, top_n, []) #cross-encoder/ms-marco-MiniLM-L-6-v2
                    for j in range(top_n):
                        new_context += f"{new_chunks[i]}\n\n"
                    
                    new_contexts.append(new_context)

        else:
            # query documents
            response = query_engine.query(query)  

            context = "Context:\n"
            chunks = []
            for i in range(top_k):
                chunk_text = response.source_nodes[i].text
                chunks.append(chunk_text)     # useful if reranking
                context += f"{chunk_text}\n\n"

            if reranking > 0:
                new_context = "Context:\n"
                new_chunks, scores, new_ids = my_reranking("BAAI/bge-reranker-base", query, chunks, top_n, query_result.ids) if reranking == 1 else my_reranking_2(llm, model, tokenizer, device, query, reference_answer, chunks, top_n, query_result.ids)

                for i in range(top_n):
                    new_context += f"{new_chunks[i]}\n\n"

    elif splitting_method in ["From scratch with Faiss vector store and SYNTACTIC splitter", "From scratch with Faiss vector store and SEMANTIC splitter"]:

        config_hash = get_config_hash(chunk_size, chunk_overlap, splitting_method, embedding_model_name)
        embeddings = load_embeddings(config_hash)

        embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)

        if splitting_method == "From scratch with Faiss vector store and SYNTACTIC splitter":
            text_parser = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                # separator=" ",
            )

            text_chunks = []
            # maintain relationship with source doc index, to help inject doc metadata 
            doc_idxs = []
            for doc_idx, doc in enumerate(document_objects):
                cur_text_chunks = text_parser.split_text(doc.text)
                text_chunks.extend(cur_text_chunks)
                doc_idxs.extend([doc_idx] * len(cur_text_chunks))
            
            if retriever_method == "base":
                nodes = []
                for idx, text_chunk in enumerate(text_chunks):
                    node = TextNode(
                        text=text_chunk,
                    )
                    src_doc = document_objects[doc_idxs[idx]]
                    node.metadata = src_doc.metadata
                    nodes.append(node)

        else:
            semantic_parser = SemanticSplitterNodeParser(
                buffer_size=1, breakpoint_percentile_threshold=85, embed_model=embed_model
            )

            nodes = semantic_parser.build_semantic_nodes_from_documents(document_objects)

        if retriever_method == "base":

            if embeddings is not None: 
                embeddings = embeddings.tolist()
                for i, emb in enumerate(embeddings):
                    nodes[i].embedding = emb
            else:
                for node in nodes:
                    node_embedding = embed_model.get_text_embedding(
                        node.get_content(metadata_mode="all")
                    )
                    node.embedding = node_embedding

                save_embeddings(nodes, config_hash)

            # create a faiss index
            d = len(nodes[0].embedding)  # dimension
            faiss_index = faiss.IndexFlatL2(d)

            vector_store = FaissVectorStore(faiss_index=faiss_index)

            vector_store.add(nodes)


        if isinstance(query, list):
            contexts = []
            new_contexts = []

            query_mode = "default"
            # query_mode = "sparse"
            # query_mode = "hybrid"

            for i in query:
                if retriever_method == "BM25":
                    corpus_tokens = bm25s.tokenize(text_chunks)
                    ret = bm25s.BM25(corpus=text_chunks)
                    ret.index(corpus_tokens)

                    query_tokens = bm25s.tokenize(qa_pairs[i-1]['question'])
                    chunks, scores = ret.retrieve(query_tokens, k=top_k)
                    chunks = chunks[0]
                    context = "Context:\n"
                    for j in range(top_k):
                        context += f"{chunks[j]}\n\n"
                    contexts.append(context)
                    #print("bm25")
                
                else:
                    query_embedding = embed_model.get_query_embedding(qa_pairs[i-1]['question'])

                    vector_store_query = VectorStoreQuery(
                        query_embedding=query_embedding, similarity_top_k=top_k, mode=query_mode
                    )
                    # returns a VectorStoreQueryResult
                    query_result = vector_store.query(vector_store_query)
                    query_result.ids = query_result.ids[::-1]

                    context = "Context:\n"
                    chunks = []
                    for j in range(top_k):
                        chunk_text = nodes[int(query_result.ids[j])].text
                        chunks.append(chunk_text)
                        context += f"{chunk_text}\n\n"
                    contexts.append(context)
                
                if retriever_evaluation:
                    _, retriever_rate = evaluate_retriever(qa_pairs[i-1]['question'], qa_pairs[i-1]['answer'], chunks, [], llm, model, tokenizer, device, get_max_new_tokens(llm, qa_pairs[i-1]['answer']))
                    
                    retriever_rates.append(retriever_rate)

                if reranking > 0:
                    new_context = "Context:\n"
                    new_chunks, scores = my_reranking("BAAI/bge-reranker-base", qa_pairs[i-1]['question'], chunks, top_n, []) if reranking == 1 else my_reranking_2(llm, model, tokenizer, device, qa_pairs[i-1]['question'], qa_pairs[i-1]['answer'], chunks, top_n, [])
                    for i in range(top_n):
                        new_context += f"{new_chunks[i]}\n\n"
                        
                    new_contexts.append(new_context)

        else:

            query_str = query

            if retriever_method == "BM25":
                text_chunk_to_idx = dict(zip(text_chunks, [i for i in range(len(text_chunks))]))
                corpus_tokens = bm25s.tokenize(text_chunks)
                ret = bm25s.BM25(corpus=text_chunks)
                ret.index(corpus_tokens)
                query_tokens = bm25s.tokenize(query_str)
                chunks, scores = ret.retrieve(query_tokens, k=top_k)
                chunks = chunks[0]
                ids = [text_chunk_to_idx[chunk] for chunk in chunks]
                context = "Context:\n"
                for i in range(top_k):
                    context += f"{chunks[i]}\n\n"
                #print("bm25")
            
            else:
                query_embedding = embed_model.get_query_embedding(query_str)

                query_mode = "default"
                # query_mode = "sparse"
                # query_mode = "hybrid"

                vector_store_query = VectorStoreQuery(
                    query_embedding=query_embedding, similarity_top_k=top_k, mode=query_mode
                )

                # returns a VectorStoreQueryResult
                query_result = vector_store.query(vector_store_query)
                query_result.ids = query_result.ids[::-1]
                ids = query_result.ids
                context = "Context:\n"

                chunks = []
                for i in range(top_k):
                    chunk_text = nodes[int(query_result.ids[i])].text
                    chunks.append(chunk_text)   # useful if reranking
                    context += f"{chunk_text}\n\n"

            if reranking > 0:
                new_context = "Context:\n"
                new_chunks, scores, new_ids = my_reranking("BAAI/bge-reranker-base", query, chunks, top_n, ids) if reranking == 1 else my_reranking_2(llm, model, tokenizer, device, query, reference_answer, chunks, top_n, ids)

                for i in range(top_n):
                    new_context += f"{new_chunks[i]}\n\n"
        
    if isinstance(query, list):
        if retriever_evaluation:
            mean_retriever_rates = {
                "DCG": np.mean([rate["DCG"] for rate in retriever_rates]),
                "NDCG": np.mean([rate["NDCG"] for rate in retriever_rates]),
                "MRR": np.mean([rate["MRR"] for rate in retriever_rates]),
                "MAP": np.mean([rate["MAP"] for rate in retriever_rates])
            }

        else:
            mean_retriever_rates = None

        return contexts, new_contexts, mean_retriever_rates
    

    return context, new_context, chunks, new_chunks, ids, new_ids


def generate_answer_with_citations(query, context, chunks, new_chunks, new_context, tokenizer, model, max_new_tokens, ids, new_ids, llm, device) -> QuotedAnswer:
    if new_context:
        context = new_context
        chunks = new_chunks
        ids = new_ids

    model.eval()

    # prompt (no additional prompt and no context)
    prompt1 = f'''[INST] {query} [/INST]'''

    messages1 = [
        {"role": "user", "content": query}
    ]

    # prompt (additional prompt and no context)
    intstructions_string = f"""You are an expert medical professional with extensive knowledge in various fields of medicine, \n
            Give short and clear answers without too many details (Give the answer directly without reformulate the query). \n
            Example1 : What is the main objective of the evidence-based guidelines for treating infections caused by MDR-GNB in hospitalized patients?\n
            To provide recommendations for treatment of infections caused by MDR-GNB in hospitalized patients.

            Don't say : The main objective of the evidence-based guidelines for treating infections caused by MDR-GNB in hospitalized patients is ..."

            Example2 : What is the estimated number of infections caused by antibiotic-resistant bacteria in Europe in 2015?\n
            600,000

            Don't say : The estimated number of infections caused by antibiotic-resistant bacteria in Europe in 2015 is 600,000.

            Please respond to the following question.
            """
    
    
    prompt_template = lambda question: f'''[INST] {intstructions_string} \n{question} \n[/INST]'''
    
    messages2 = [
        {"role": "system", "content": intstructions_string},
        {"role": "user", "content": query}
    ]

    prompt2 =  prompt_template(query)

    # prompt (no additional prompt and context)
    prompt_template_w_context = lambda context, question: f"""[INST]
    {context}
    Please respond to the following question. Use the context above if it is helpful.

    {question}
    [/INST]
    """
    
    messages3 = [
        {"role": "system", "content": f"{context}\nPlease respond to the following question. Use the context above if it is helpful."},
        {"role": "user", "content": query}
    ]

    prompt3 = prompt_template_w_context(context, query)

    # prompt (context and additional prompt)
    prompt_template_w_context = lambda context, question: f"""[INST]You are an expert medical professional with extensive knowledge in various fields of medicine, \n
    Give short and clear answers without too many details (Give the answer directly without reformulate the query). \n
    Example1 : What is the main objective of the evidence-based guidelines for treating infections caused by MDR-GNB in hospitalized patients?\n
    To provide recommendations for treatment of infections caused by MDR-GNB in hospitalized patients.

    Don't say : The main objective of the evidence-based guidelines for treating infections caused by MDR-GNB in hospitalized patients is ..."

    Example2 : What is the estimated number of infections caused by antibiotic-resistant bacteria in Europe in 2015?\n
    600,000

    Don't say : The estimated number of infections caused by antibiotic-resistant bacteria in Europe in 2015 is 600,000.


    {context}
    Please respond to the following question. Use the context above if it is helpful.

    {question}
    [/INST]
    """
    
    messages4 = [
        {"role": "system", "content": f"You are an expert medical professional with extensive knowledge in various fields of medicine. You are here to provide accurate, evidence-based information and guidance on a wide range of medical topics. Preferably give clear answers without too many details.\n\n {context}\nPlease respond to the following question. Use the context above if it is helpful."},
        {"role": "user", "content": query}
    ]

    prompt4 = prompt_template_w_context(context, query)

    prompts = {
        1: prompt1,
        2: prompt2,
        3: prompt3,
        4: prompt4
    }

    messages = {
        1: messages1,
        2: messages2,
        3: messages3,
        4: messages4
    }

    responses = {}

    if llm == "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ":
        for i in range(1, 5):
            inputs = tokenizer(prompts[i], return_tensors="pt").to(device)

            outputs = model.generate(
                input_ids=inputs["input_ids"],#.to("cuda"),
                max_new_tokens=max_new_tokens +50,  
                num_return_sequences=1,  
                pad_token_id=tokenizer.eos_token_id,  
                eos_token_id=tokenizer.eos_token_id,
                #early_stopping=True,  # Arrête la génération dès qu'un token de fin est atteint
                #num_beams=5,  # Utilisation de la recherche par faisceaux
                #no_repeat_ngram_size=2,  # Empêche la répétition des bigrammes
                length_penalty=2.0  # Pénalité de longueur pour favoriser des séquences plus courtes
            )


            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            response = generated_text[len(prompts[i]):].strip()

            citations = []
            for j in range(len(chunks)):
                chunk_idx = int(ids[j])
                
                citation = Citation(
                    source_id=chunk_idx,
                    quote=chunks[j]
                )
                citations.append(citation)

            quoted_answer = QuotedAnswer(
                answer=response,
                citations=citations
            )

            responses[i] = quoted_answer
    else:
        pipe = pipeline( 
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            device=device.index if device.type == 'cuda' else -1
        ) 

        generation_args = { 
            "max_new_tokens": max_new_tokens+50, 
            "return_full_text": False, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        for i in range(1, 5):
            output = pipe(messages[i], **generation_args) 
            response = output[0]['generated_text']

            citations = []
            for j in range(len(chunks)):
                chunk_idx = int(ids[j])
                
                citation = Citation(
                    source_id=chunk_idx,
                    quote=chunks[j]
                )
                citations.append(citation)

            quoted_answer = QuotedAnswer(
                answer=response,
                citations=citations
            )

            responses[i] = quoted_answer

    return responses

def generate_answers(query, contexts, new_contexts, tokenizer, model, llm, qa_pairs, device):
    if new_contexts:
        contexts = new_contexts

    model.eval()

    prompts1 = []
    prompts2 = []
    prompts3 = []
    prompts4 = []

    messages1 = []
    messages2 = []
    messages3 = []
    messages4 = []

    for i, q in enumerate(query):
        # prompt (no additional prompt and no context)
        prompt1 = f'''[INST] {qa_pairs[q-1]['question']} [/INST]'''
        prompts1.append(prompt1)

        message1 = [
            {"role": "user", "content": qa_pairs[q-1]['question']}
        ]
        messages1.append(message1)

        # prompt (additional prompt and no context)
        intstructions_string = f"""You are an expert medical professional with extensive knowledge in various fields of medicine, \n
            Give short and clear answers without too many details (Give the answer directly without reformulate the query). \n
            Example1 : What is the main objective of the evidence-based guidelines for treating infections caused by MDR-GNB in hospitalized patients?\n
            To provide recommendations for treatment of infections caused by MDR-GNB in hospitalized patients.

            Don't say : The main objective of the evidence-based guidelines for treating infections caused by MDR-GNB in hospitalized patients is ..."

            Example2 : What is the estimated number of infections caused by antibiotic-resistant bacteria in Europe in 2015?\n
            600,000

            Don't say : The estimated number of infections caused by antibiotic-resistant bacteria in Europe in 2015 is 600,000.

            Please respond to the following question.
            """
        
        
        prompt_template = lambda question: f'''[INST] {intstructions_string} \n{question} \n[/INST]'''
        
        message2 = [
            {"role": "system", "content": intstructions_string},
            {"role": "user", "content": qa_pairs[q-1]['question']}
        ]
        messages2.append(message2)

        prompt2 =  prompt_template(qa_pairs[q-1]['question'])
        prompts2.append(prompt2)

        # prompt (no additional prompt and context)
        prompt_template_w_context = lambda context, question: f"""[INST]
        {context}
        Please respond to the following question. Use the context above if it is helpful.

        {question}
        [/INST]
        """
        
        message3 = [
            {"role": "system", "content": f"{contexts[i]}\nPlease respond to the following question. Use the context above if it is helpful."},
            {"role": "user", "content": qa_pairs[q-1]['question']}
        ]
        messages3.append(message3)

        prompt3 = prompt_template_w_context(contexts[i], qa_pairs[q-1]['question'])
        prompts3.append(prompt3)

        # prompt (context and additional prompt)
        prompt_template_w_context = lambda context, question: f"""[INST]You are an expert medical professional with extensive knowledge in various fields of medicine, \n
        Give short and clear answers without too many details (Give the answer directly without reformulate the query). \n
        Example1 : What is the main objective of the evidence-based guidelines for treating infections caused by MDR-GNB in hospitalized patients?\n
        To provide recommendations for treatment of infections caused by MDR-GNB in hospitalized patients.

        Don't say : The main objective of the evidence-based guidelines for treating infections caused by MDR-GNB in hospitalized patients is ...

        Example2 : What is the estimated number of infections caused by antibiotic-resistant bacteria in Europe in 2015?\n
        600,000

        Don't say : The estimated number of infections caused by antibiotic-resistant bacteria in Europe in 2015 is 600,000.


        {context}
        Please respond to the following question. Use the context above if it is helpful.

        {question}
        [/INST]
        """
        
        message4 = [
            {"role": "system", "content": f"You are an expert medical professional with extensive knowledge in various fields of medicine.\n Give short and clear answers without too many details (Give the answer directly without reformulate the query).\n Example1 : What is the main objective of the evidence-based guidelines for treating infections caused by MDR-GNB in hospitalized patients?\n To provide recommendations for treatment of infections caused by MDR-GNB in hospitalized patients.\n\n Don't say : The main objective of the evidence-based guidelines for treating infections caused by MDR-GNB in hospitalized patients is ...\n\n Example2 : What is the estimated number of infections caused by antibiotic-resistant bacteria in Europe in 2015?\n 600,000\n\n Don't say : The estimated number of infections caused by antibiotic-resistant bacteria in Europe in 2015 is 600,000.\n\n {contexts[i]}\n Please respond to the following question. Use the context above if it is helpful."},
            {"role": "user", "content": qa_pairs[q-1]['question']}
        ]
        messages4.append(message4)

        prompt4 = prompt_template_w_context(contexts[i], qa_pairs[q-1]['question'])
        prompts4.append(prompt4)

        prompts = {
            1: prompts1,
            2: prompts2,
            3: prompts3,
            4: prompts4
        }

        messages = {
            1: messages1,
            2: messages2,
            3: messages3,
            4: messages4
        }

    responses = {}

    if llm == "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ":
        for i in range(1, 5):
            responses_case = []
            for j in range(len(prompts1)):
                inputs = tokenizer(prompts[i][j], return_tensors="pt").to(device)

                max_new_tokens = get_max_new_tokens(llm, qa_pairs[query[j]-1]['answer'])

                outputs = model.generate(
                    input_ids=inputs["input_ids"],#.to("cuda"),
                    max_new_tokens=max_new_tokens +50, 
                    num_return_sequences=1,  
                    pad_token_id=tokenizer.eos_token_id, 
                    eos_token_id=tokenizer.eos_token_id,
                    length_penalty=2,
                    #num_beams=10,
                    #temperature=0.3
                )

                generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                response = generated_text[len(prompts[i][j]):].strip()

                responses_case.append(response)

            responses[i] = responses_case
    else:
        pipe = pipeline( 
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            device=device.index if device.type == 'cuda' else -1,
        ) 
    
        for i in range(1, 5):
            responses_case = []
            for j in range(len(messages1)):
                max_new_tokens = get_max_new_tokens(llm, qa_pairs[query[j]-1]['answer']) 
                generation_args = { 
                    "max_new_tokens": max_new_tokens +50, 
                    "return_full_text": False, 
                    "temperature": 0.0, 
                    "do_sample": False, 
                    #"num_beams": 10,
                    #"length_penalty": 2,
                }

                output = pipe(messages[i][j], **generation_args) 
                response = output[0]['generated_text']

                responses_case.append(response)

            responses[i] = responses_case

    return responses

def results_w_metrics(responses, context, reference_answer, reference_context, new_context):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    metrics_score = {}

    context_similarity = []
    context_meteor_score = meteor_score([reference_context.split()], context.split())
    context_similarity.append(context_meteor_score)
    context_bertscore = []
    P, R, F1 = score([context], [reference_context], lang='en', rescale_with_baseline=True)
    context_bertscore.append(round(P.mean().item(), 3))
    context_bertscore.append(round(R.mean().item(), 3))
    context_bertscore.append(round(F1.mean().item(), 3))
    context_similarity.append(context_bertscore)
    metrics_score[0] = context_similarity

    if new_context:
        new_context_similarity = []
        new_context_meteor_score = meteor_score([reference_context.split()], new_context.split())
        new_context_similarity.append(new_context_meteor_score)
        new_context_bertscore = []
        P, R, F1 = score([new_context], [reference_context], lang='en', rescale_with_baseline=True)
        new_context_bertscore.append(round(P.mean().item(), 3))
        new_context_bertscore.append(round(R.mean().item(), 3))
        new_context_bertscore.append(round(F1.mean().item(), 3))
        new_context_similarity.append(new_context_bertscore)
        metrics_score[5] = new_context_similarity

    for i in range(1, 5):
        metrics_score[i] = {}

        bleu_score = {}
        bleu_score["1_gram"] = sentence_bleu(reference_answer.split(), responses[i].answer, weights=(1, 0, 0, 0))
        bleu_score["2_gram"] = sentence_bleu(reference_answer.split(), responses[i].answer, weights=(0, 1, 0, 0))
        bleu_score["3_gram"] = sentence_bleu(reference_answer.split(), responses[i].answer, weights=(0, 0, 1, 0))
        bleu_score["4_gram"] = sentence_bleu(reference_answer.split(), responses[i].answer, weights=(0, 0, 0, 1))
    
        rouge_score = {}
        rouge_scores = scorer.score(reference_answer, responses[i].answer)
        rouge_score["rouge_1"] = {}
        rouge_score["rouge_1"]["precision"] = rouge_scores['rouge1'].precision
        rouge_score["rouge_1"]["recall"] = rouge_scores['rouge1'].recall
        rouge_score["rouge_1"]["fmeasure"] = rouge_scores['rouge1'].fmeasure
        rouge_score["rouge_L"] = {}
        rouge_score["rouge_L"]["precision"] = rouge_scores['rougeL'].precision
        rouge_score["rouge_L"]["recall"] = rouge_scores['rougeL'].recall
        rouge_score["rouge_L"]["fmeasure"] = rouge_scores['rougeL'].fmeasure

        m_score = meteor_score([reference_answer.split()], responses[i].answer.split())
        
        bertscore = {}
        P, R, F1 = score([responses[i].answer], [reference_answer], lang='en', rescale_with_baseline=True)
        bertscore["Precision"] = round(P.mean().item(), 3)
        bertscore["Recall"] = round(R.mean().item(), 3)
        bertscore["F1_measure"] = round(F1.mean().item(), 3)

        metrics_score[i]["BLEU"] = bleu_score
        metrics_score[i]["ROUGE"] = rouge_score
        metrics_score[i]["METEOR"] = m_score
        metrics_score[i]["BERTScore"] = bertscore

    return metrics_score


def results_for_many_queries(candidates, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    metrics_score = {}

    for j in range(1, 5):
        metrics_score[j] = {}
    
        bleu_score = {}
        ind_1_gram = [sentence_bleu(references[i].split(), candidates[j][i], weights=(1, 0, 0, 0)) for i in range(len(references))]
        ind_2_gram = [sentence_bleu(references[i].split(), candidates[j][i], weights=(0, 1, 0, 0)) for i in range(len(references))]
        ind_3_gram = [sentence_bleu(references[i].split(), candidates[j][i], weights=(0, 0, 1, 0)) for i in range(len(references))]
        ind_4_gram = [sentence_bleu(references[i].split(), candidates[j][i], weights=(0, 0, 0, 1)) for i in range(len(references))]

        bleu_score["1_gram"] = np.mean(ind_1_gram)
        bleu_score["2_gram"] = np.mean(ind_2_gram)
        bleu_score["3_gram"] = np.mean(ind_3_gram)
        bleu_score["4_gram"] = np.mean(ind_4_gram)

        rouge_score = {}
        scores = [scorer.score(references[i], candidates[j][i]) for i in range(len(references))]

        precision_rouge_1 = [scores[i]['rouge1'][0] for i in range(len(scores))]
        recall_rouge_1 = [scores[i]['rouge1'][1] for i in range(len(scores))]
        fmeasure_rouge_1 = [scores[i]['rouge1'][2] for i in range(len(scores))]

        precision_rouge_L = [scores[i]['rougeL'][0] for i in range(len(scores))]
        recall_rouge_L = [scores[i]['rougeL'][1] for i in range(len(scores))]
        fmeasure_rouge_L = [scores[i]['rougeL'][2] for i in range(len(scores))]
    
        rouge_score["rouge_1"] = {}
        rouge_score["rouge_1"]["precision"] = np.mean(precision_rouge_1)
        rouge_score["rouge_1"]["recall"] = np.mean(recall_rouge_1)
        rouge_score["rouge_1"]["fmeasure"] = np.mean(fmeasure_rouge_1)
        rouge_score["rouge_L"] = {}
        rouge_score["rouge_L"]["precision"] = np.mean(precision_rouge_L)
        rouge_score["rouge_L"]["recall"] = np.mean(recall_rouge_L)
        rouge_score["rouge_L"]["fmeasure"] = np.mean(fmeasure_rouge_L)
    

        m_scores = [meteor_score([references[i].split()], candidates[j][i].split()) for i in range(len(references))]
        m_score = np.mean(m_scores)


        bertscore = {}
        bert_scores = [score([candidates[j][i]], [references[i]], lang='en', rescale_with_baseline=True) for i in range(len(references))]
    
        P = [float(bert_scores[i][0]) for i in range(len(bert_scores))]
        R = [float(bert_scores[i][1]) for i in range(len(bert_scores))]
        F1 = [float(bert_scores[i][2]) for i in range(len(bert_scores))]

        bertscore["Precision"] = round(np.mean(P), 3)
        bertscore["Recall"] = round(np.mean(R), 3)
        bertscore["F1_measure"] = round(np.mean(F1), 3)

        metrics_score[j]["BLEU"] = bleu_score
        metrics_score[j]["ROUGE"] = rouge_score
        metrics_score[j]["METEOR"] = m_score
        metrics_score[j]["BERTScore"] = bertscore

    return metrics_score


def evaluate_retriever(query, reference_answer, chunks, new_chunks, llm, model, tokenizer, device, max_new_tokens):
    responses = []

    prompt_template_w_context = lambda context, question: f"""[INST]
    {context}
    Please respond to the following question. Use the context above if it is helpful.

    {question}
    [/INST]
    """

    for chunk in chunks:
        if llm == "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ":
            prompt = prompt_template_w_context(chunk, query)

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                input_ids=inputs["input_ids"],#.to("cuda"),
                max_new_tokens=max_new_tokens +50, 
                num_return_sequences=1,  
                pad_token_id=tokenizer.eos_token_id, 
                eos_token_id=tokenizer.eos_token_id,
                #length_penalty=2,
                #num_beams=10,
                #temperature=0.3
                length_penalty=2.0
            )

            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            response = generated_text[len(prompt):].strip()
            responses.append(response)

        else:
            message = [
            {"role": "system", "content": f"{chunk}\nPlease respond to the following question. Use the context above if it is helpful."},
            {"role": "user", "content": query}
            ]

            pipe = pipeline( 
                "text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                device=device.index if device.type == 'cuda' else -1,
            ) 

            generation_args = { 
                "max_new_tokens": max_new_tokens +50, 
                "return_full_text": False, 
                "temperature": 0.0, 
                "do_sample": False, 
                #"num_beams": 10,
                #"length_penalty": 2,
            }

            output = pipe(message, **generation_args) 
            response = output[0]['generated_text']
            responses.append(response)

    chunk_scores = []
    for response in responses:
        m_score = round(meteor_score([reference_answer.split()], response.split()), 3)
        chunk_scores.append(m_score)
    chunk_retriever = {}
    chunk_retriever["DCG"] = compute_DCG(chunk_scores, len(chunk_scores))
    chunk_retriever["NDCG"] = compute_NDCG(chunk_scores, len(chunk_scores))
    chunk_retriever["MRR"] = compute_MRR(chunk_scores, len(chunk_scores))
    chunk_retriever["MAP"] = compute_MAP(chunk_scores, len(chunk_scores))

    if new_chunks: # reranking is used
        chunk_to_score = dict(zip(chunks, chunk_scores))
        new_chunk_scores = [chunk_to_score[chunk] for chunk in new_chunks]
        ideal_new_chunk_scores = sorted(chunk_scores, reverse=True)[:len(new_chunk_scores)]

        scores = (chunk_scores, new_chunk_scores, ideal_new_chunk_scores)

    else:
        scores = (chunk_scores,)
        
    retriever_rate = chunk_retriever

    return scores, retriever_rate


def prompt_evaluation_query(query, predicted_answer, context, new_context):
    if new_context:
        context = new_context

    modelfile = '''
    FROM mixtral:8x7b
    PARAMETER temperature 0.1
    '''

    ollama.create(model='example', modelfile=modelfile)

    model_name = "example"

    eval_metrics = {}

    #Faithfulness
    prompt_template_1 = lambda question, answer: f"""
    Given a question and answer, create one or more statements from each sentence in the given answer.
    question: {question}
    answer: {answer}
    """

    prompt_1 = prompt_template_1(query, predicted_answer)

    response = ollama.chat(model=model_name, messages=[
    {
        'role': 'user',
        'content': prompt_1,
    },
    ])

    text = response['message']['content']
    statements = text.split('\n')
    statements = [re.sub(r'^\s*\d+\.\s*', '', statement) for statement in statements]

    prompt_template_2 = lambda context, statements: f"""
    {context}
    Consider the given context and following statements, then determine whether they are supported by the information present in the context. Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.
    """ + '\n'.join([f"statement: {statement}\nAnswer: Yes/No\nExplanation: ..." for statement in statements]) + '\n'

    prompt_2 = prompt_template_2(context, statements)

    response = ollama.chat(model=model_name, messages=[
    {
        'role': 'user',
        'content': prompt_2,
    },
    ])

    text = response['message']['content']

    answers = re.findall(r'Answer: (Yes|No)', text)
    nb_yes = answers.count('Yes')

    if len(answers) == 0:
        faithfulness = -1
    else:
        faithfulness = nb_yes/len(answers)

    eval_metrics["faithfulness"] = faithfulness

    #Answer relevance
    prompt_template_3 = lambda n, answer: f"""
    Generate {n} questions for the given answer.
    answer: {answer}
    """

    prompt_3 = prompt_template_3(5, predicted_answer)

    response = ollama.chat(model=model_name, messages=[
    {
        'role': 'user',
        'content': prompt_3,
    },
    ])

    text = response['message']['content']
    questions = text.split('\n')
    questions = [re.sub(r'^\s*\d+\.\s*', '', question) for question in questions]
    
    query_embeddings = ollama.embeddings(model=model_name, prompt=query)['embedding']
    question_embeddings = [ollama.embeddings(model=model_name, prompt=question)['embedding'] for question in questions]
    query_embeddings = np.array(query_embeddings)
    question_embeddings = [np.array(q_embeddings) for q_embeddings in question_embeddings]

    similarities = [cosine_similarity([query_embeddings], [q_embeddings]) for q_embeddings in question_embeddings]
    similarities = [similarity[0][0] for similarity in similarities]
    
    answer_relevance = np.mean(similarities)
    eval_metrics["answer relevance"] = answer_relevance

    #Context relevance
    prompt_template_4 = lambda question, context: f"""
    You must extract only the **exact sentences** from the provided context that directly help answer the following question: {question}. **Do not rephrase, summarize, or provide explanations.** Only return sentences as they are written in the context. If no exact sentences are relevant, return the phrase "Insufficient Information".

    {context}
    """
    context = context.replace('\n', '')

    prompt_4 = prompt_template_4(query, context)

    response = ollama.chat(model=model_name, messages=[
    {
        'role': 'user',
        'content': prompt_4,
    },
    ])

    text = response['message']['content']
    sentences_context = split_into_sentences(context)
    sentences_context[0] = sentences_context[0].replace("Context:", "")
    sentences_text = split_into_sentences(text)
    if "Insufficient Information" in text:
        context_relevance = 0

    elif all(sentence not in sentences_text for sentence in sentences_context):
        context_relevance = -1

    else:
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        number_of_sentences_text = len(sentences)

        sentences = re.split(r'[.!?]', context)
        sentences = [s.strip() for s in sentences if s.strip()]
        number_of_sentences_context = len(sentences)

        context_relevance = number_of_sentences_text/number_of_sentences_context

    eval_metrics["context relevance"] = context_relevance

    return eval_metrics


def prompt_evaluation_queries(query, qa_pairs, responses, contexts, new_contexts):
    if new_contexts:
        contexts = new_contexts

    modelfile = '''
    FROM mixtral:8x7b
    PARAMETER temperature 0.1
    '''

    ollama.create(model='example', modelfile=modelfile)

    model_name = "example"

    eval_metrics = {}
    faithfulness_list = []
    answer_relevance_list = []
    context_relevance_list = []

    for i, q in enumerate(query):
        #Faithfulness
        prompt_template_1 = lambda question, answer: f"""
        Given a question and answer, create one or more statements from each sentence in the given answer.
        question: {question}
        answer: {answer}
        """

        prompt_1 = prompt_template_1(qa_pairs[q-1]['question'], responses[i])

        response = ollama.chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': prompt_1,
        },
        ])

        text = response['message']['content']
        statements = text.split('\n')
        statements = [re.sub(r'^\s*\d+\.\s*', '', statement) for statement in statements]

        prompt_template_2 = lambda context, statements: f"""
        {context}
        Consider the given context and following statements, then determine whether they are supported by the information present in the context. Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.
        """ + '\n'.join([f"statement: {statement}\nAnswer: Yes/No\nExplanation: ..." for statement in statements]) + '\n'

        prompt_2 = prompt_template_2(contexts[i], statements)

        response = ollama.chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': prompt_2,
        },
        ])

        text = response['message']['content']

        answers = re.findall(r'Answer: (Yes|No)', text)
        nb_yes = answers.count('Yes')

        if len(answers) == 0:
            faithfulness = -1

        else:
            faithfulness = nb_yes/len(answers)

        faithfulness_list.append(faithfulness)

        #Answer relevance
        prompt_template_3 = lambda n, answer: f"""
        Generate {n} questions for the given answer.
        answer: {answer}
        """

        prompt_3 = prompt_template_3(5, responses[i])

        response = ollama.chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': prompt_3,
        },
        ])

        text = response['message']['content']
        questions = text.split('\n')
        questions = [re.sub(r'^\s*\d+\.\s*', '', question) for question in questions]
        
        query_embeddings = ollama.embeddings(model=model_name, prompt=qa_pairs[q-1]['question'])['embedding']
        question_embeddings = [ollama.embeddings(model=model_name, prompt=question)['embedding'] for question in questions]
        query_embeddings = np.array(query_embeddings)
        question_embeddings = [np.array(q_embeddings) for q_embeddings in question_embeddings]

        similarities = [cosine_similarity([query_embeddings], [q_embeddings]) for q_embeddings in question_embeddings]
        similarities = [similarity[0][0] for similarity in similarities]
        
        answer_relevance = np.mean(similarities)
        answer_relevance_list.append(answer_relevance)

        #Context relevance
        prompt_template_4 = lambda question, context: f"""
        You must extract only the **exact sentences** from the provided context that directly help answer the following question: {question}. **Do not rephrase, summarize, or provide explanations.** Only return sentences as they are written in the context. If no exact sentences are relevant, return the phrase "Insufficient Information".

        {context}
        """
        contexts[i] = contexts[i].replace('\n', '')

        prompt_4 = prompt_template_4(qa_pairs[q-1]['question'], contexts[i])

        response = ollama.chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': prompt_4,
        },
        ])

        text = response['message']['content']
        sentences_context = split_into_sentences(contexts[i])
        sentences_context[0] = sentences_context[0].replace("Context:", "")
        sentences_text = split_into_sentences(text)
        if "Insufficient Information" in text:
            context_relevance = 0

        elif all(sentence not in sentences_text for sentence in sentences_context):
            context_relevance = -1

        else:
            sentences = re.split(r'[.!?]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            number_of_sentences_text = len(sentences)

            sentences = re.split(r'[.!?]', contexts[i])
            sentences = [s.strip() for s in sentences if s.strip()]
            number_of_sentences_context = len(sentences)

            context_relevance = number_of_sentences_text/number_of_sentences_context

        context_relevance_list.append(context_relevance)

    if faithfulness_list.count(-1) > len(faithfulness_list)/3:
        eval_metrics["faithfulness"] = -1
    else:
        faithfulness_list = [x for x in faithfulness_list if x != -1]
        eval_metrics["faithfulness"] = np.mean(faithfulness_list)  
    eval_metrics["answer relevance"] = np.mean(answer_relevance_list)
    if context_relevance_list.count(-1) > len(context_relevance_list)/3:
        eval_metrics["context relevance"] = -1
    else:
        context_relevance_list = [x for x in context_relevance_list if x != -1]
        eval_metrics["context relevance"] = np.mean(context_relevance_list)

    return eval_metrics


def evaluate_model(embedding, llm, splitting_method, chunksize, overlap, top_k, reranking, top_n, documents, query, reference_answer, reference_context, results, qa_pairs, retriever_method, retriever_evaluation, is_automated_evaluation):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = AutoModelForCausalLM.from_pretrained(llm,
                                                device_map="cuda",  
                                                torch_dtype="auto",  
                                                trust_remote_code=True,)
                                                # device_map="auto",
                                                # trust_remote_code=True,
                                                 # revision="main")

    model = model.to(device)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm, use_fast=True)

    if isinstance(query, list):
        contexts, new_contexts, mean_retriever_rates = splitting_and_context(splitting_method, embedding, chunksize, overlap, documents, top_k, query, reference_answer, qa_pairs, reranking, top_n, llm, model, tokenizer, device, retriever_method, retriever_evaluation)

        responses = generate_answers(query, contexts, new_contexts, tokenizer, model, llm, qa_pairs, device)

        metrics = results_for_many_queries(responses, reference_answer)

        if is_automated_evaluation:
            automated_evaluation = prompt_evaluation_queries(query, qa_pairs, responses[3], contexts, new_contexts)

        retriever_rate_entry = {}
        if retriever_evaluation:
            retriever_rate_entry = {
                "retriever rate": {
                    "DCG": mean_retriever_rates["DCG"],
                    "NDCG": mean_retriever_rates["NDCG"],
                    "MRR": mean_retriever_rates["MRR"],
                    "MAP": mean_retriever_rates["MAP"]
                },
            }

        automated_evaluation_entry = {}
        if is_automated_evaluation:
            automated_evaluation_entry = {
                "automated evaluation": {
                    "faithfulness": automated_evaluation["faithfulness"],
                    "answer relevance": automated_evaluation["answer relevance"],
                    "context relevance": automated_evaluation["context relevance"]
                },
            }

        entry = OrderedDict({
            "parameters": {
                "embedding model": embedding,
                "llm": llm,
                "splitting method": splitting_method,
                "retriever method": retriever_method,
                "retriever evaluation": retriever_evaluation,
                "chunk size": chunksize,
                "overlap": overlap,
                "top_k": top_k,
                "re-ranking": reranking,
                "top_n": top_n
            },
            **retriever_rate_entry,
            "queries": [qa_pairs[i-1]['question'] for i in query],
            "reference answers": reference_answer,
            "predicted answers": {
                "no additional prompt / no context": responses[1],
                "additional prompt / no context": responses[2],
                "no additional prompt / context": responses[3],
                "additional prompt / context": responses[4],
            },
            **automated_evaluation_entry,
            "metrics": {
                "no additional prompt / no context": metrics[1],
                "additional prompt / no context": metrics[2],
                "no additional prompt / context": metrics[3],
                "additional prompt / context": metrics[4],
            }
        })

        results.append(entry)

    else:
        context, new_context, chunks, new_chunks, ids, new_ids = splitting_and_context(splitting_method, embedding, chunksize, overlap, documents, top_k, query, reference_answer, qa_pairs, reranking, top_n, llm, model, tokenizer, device, retriever_method, retriever_evaluation)

        max_new_tokens = get_max_new_tokens(llm, get_answer_by_question(query, qa_pairs))

        if retriever_evaluation:
            retriever_score, retriever_rate = evaluate_retriever(query, reference_answer, chunks, new_chunks, llm, model, tokenizer, device, max_new_tokens)

        responses = generate_answer_with_citations(query, context, chunks, new_chunks, new_context, tokenizer, model, max_new_tokens, ids, new_ids, llm, device)  

        metrics = results_w_metrics(responses, context, reference_answer, reference_context, new_context)

        if is_automated_evaluation:
            automated_evaluation = prompt_evaluation_query(query, responses[3].answer, context, new_context)

        retriever_rate_entry = {}
        if retriever_evaluation:
            retriever_rate_entry = {
                "retriever rate": {
                    "DCG": retriever_rate["DCG"],
                    "NDCG": retriever_rate["NDCG"],
                    "MRR": retriever_rate["MRR"],
                    "MAP": retriever_rate["MAP"]
                },
            }
    
        automated_evaluation_entry = {}
        if is_automated_evaluation:
            automated_evaluation_entry = {
                "automated evaluation": {
                    "faithfulness": automated_evaluation["faithfulness"],
                    "answer relevance": automated_evaluation["answer relevance"],
                    "context relevance": automated_evaluation["context relevance"]
                },
            }

        entry = OrderedDict({
            "parameters": {
                "embedding model": embedding,
                "llm": llm,
                "splitting method": splitting_method,
                "retriever method": retriever_method,
                "retriever evaluation": retriever_evaluation,
                "chunk size": chunksize,
                "overlap": overlap,
                "top_k": top_k,
                "max new tokens": max_new_tokens + 50,
                "re-ranking": reranking,
                "top_n": top_n
            },
            "query": query,
            "reference context": reference_context
        })

        entry["chunks retrieved"] = {}
        for i, chunk in enumerate(chunks, 1):
            entry["chunks retrieved"][f"chunk {i}"] = chunk

        if retriever_evaluation:
            entry["retriever evaluation"] = {}
            for i, score in enumerate(retriever_score[0], 1):
                entry["retriever evaluation"][f"chunk {i}"] = score

        entry.update({
            **retriever_rate_entry,
            "context": context,
            "context similarity": {
                "METEOR": metrics[0][0],
                "BERTScore": {
                    "Precision": metrics[0][1][0],
                    "Recall": metrics[0][1][1],
                    "F1_measure": metrics[0][1][2]
                    }
            }
        })

        if reranking > 0:
            entry["new chunks retrieved"] = {}
            for i, chunk in enumerate(new_chunks, 1):
                entry["new chunks retrieved"][f"new chunk {i}"] = chunk
            if retriever_evaluation:
                entry["new retriever evaluation"] = {}
                for i, score in enumerate(retriever_score[1], 1):
                    entry["new retriever evaluation"][f"new chunk {i}"] = score
                entry["ideal new retriever evaluation"] = {}
                for i, score in enumerate(retriever_score[2], 1):
                    entry["ideal new retriever evaluation"][f"new chunk {i}"] = score
            entry["new context after re-ranking"] = new_context
            entry["new context similarity"] = {
                "METEOR": metrics[5][0],
                "BERTScore": {
                    "Precision": metrics[5][1][0],
                    "Recall": metrics[5][1][1],
                    "F1_measure": metrics[5][1][2]
                }
            }

        entry.update({
            **automated_evaluation_entry,
            "reference answer": reference_answer,
            "predicted answer": {
                "no additional prompt / no context": responses[1].answer,
                "additional prompt / no context": responses[2].answer,
                "no additional prompt / context": responses[3].answer,
                "additional prompt / context": responses[4].answer,
            },
        })

        entry["citations"] = {}
        for i, citation in enumerate(responses[1].citations, 1):
            entry["citations"][i] = {
                "source_id": citation.source_id, 
                "quote": citation.quote
            }

        entry.update({
            "metrics": {
                "no additional prompt / no context": metrics[1],
                "additional prompt / no context": metrics[2],
                "no additional prompt / context": metrics[3],
                "additional prompt / context": metrics[4],
            }
        })

        results.append(entry)



if __name__ == "__main__":

    if torch.cuda.is_available():
        print("CUDA is available. You have", torch.cuda.device_count(), "GPUs.")
        print("CUDA version:", torch.version.cuda)
    else:
        print("CUDA is not available.")

    # load the JSON evaluation file
    with open('../data/assessment_set/dataset.json', 'r') as dataset:
        qa_pairs = json.load(dataset)
    set_size = len(qa_pairs)

    # extract data from HTML files
    documents = HTMLDirectoryReader("../data/Medical_Guidelines/HTML").load_data()
        
    # Transforming text strings into Document objects
    document_objects = [Document(text=doc, title=f"Document {i+1}") for i, doc in enumerate(documents)]

    options_query = [qa_pair['question'] for qa_pair in qa_pairs]
    
    config_directory_path = "config_files/"
    options_config = glob.glob(config_directory_path + "*.ini")

    options_thread_optimization = ["Yes", "No"]
    #thread_optimization = get_choice('thread optimization', options_thread_optimization)
    thread_optimization = "No"

    #query = get_choice('query', options_query)
    # if is_unique_list_of_ints(query):
    #     query = ast.literal_eval(query)
    #query = [9, 12, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165]
    query = [9, 12, 32, 43, 44, 52, 54, 61, 66, 70, 78, 85, 87, 89, 90, 93, 97, 101, 104, 109, 111, 117, 118, 123, 125, 126, 132, 138, 140, 142, 147, 150, 152, 153, 159, 161, 164]
    #query = [9, 12, 46, 76, 93, 123, 159]
    #query = list(range(1, 166))
    #query = "How is the evidence on treatment of infections caused by 3GCephRE organized?"

    if isinstance(query, list):
        reference_context = ""
        reference_answer = [qa_pairs[i-1]['answer'] for i in query]
    else:
        reference_context = get_context_by_question(query, qa_pairs)
        reference_answer = get_answer_by_question(query, qa_pairs)

    #config_name = get_choice('config', options_config)
    config_name = "config_files/chunksize_config.ini"

    config = configparser.ConfigParser()

    config.sections()

    config.read(config_name)

    # Get params from config 
    embedding = eval(config.get('models', 'embedding'))
    llm = eval(config.get('models', 'LLM'))
    splitting_method = eval(config.get('splitting', 'method'))
    retriever_method = eval(config.get('retriever', 'method'))
    retriever_evaluation = eval(config.get('retriever', 'retriever_evaluation'))
    chunksize = eval(config.get('splitting', 'chunksizes'))
    overlap = eval(config.get('splitting', 'overlap'))
    top_k = eval(config.get('parameters', 'top_k'))
    reranking = eval(config.get('reranking', 'used'))
    top_n = eval(config.get('reranking', 'top_n'))
    automated_evaluation = eval(config.get('parameters', 'automated_evaluation'))

    paramlist = [embedding, llm, splitting_method, retriever_method, retriever_evaluation, chunksize, overlap, top_k, reranking, top_n, automated_evaluation]

    results = []
    if thread_optimization == "No":

        for test in itertools.product(*paramlist):
            embedding = test[0]
            llm = test[1]
            splitting_method = test[2]
            retriever_method = test[3]
            retriever_evaluation = test[4]
            chunksize = test[5]
            overlap = test[6]
            top_k = test[7]
            reranking = test[8]
            top_n = test[9]
            automated_evaluation = test[10]
            evaluate_model(embedding, llm, splitting_method, chunksize, overlap, top_k, reranking, top_n, document_objects, query, reference_answer, reference_context, results, qa_pairs, retriever_method, retriever_evaluation, automated_evaluation)

    else:
        # Create a lock
        # lock = threading.Lock()

        # Using ThreadPoolExecutor for parallel execution
        #with ThreadPoolExecutor() as executor:
        multiprocessing.set_start_method('spawn', force=True)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for test in itertools.product(*paramlist):
                embedding = test[0]
                llm = test[1]
                splitting_method = test[2]
                retriever_method = test[3]
                retriever_evaluation = test[4]
                chunksize = test[5]
                overlap = test[6]
                top_k = test[7]
                reranking = test[8]
                top_n = test[9]
                automated_evaluation = test[10]
                futures.append(executor.submit(evaluate_model, embedding, llm, splitting_method, chunksize, overlap, top_k, reranking, top_n, document_objects, query, reference_answer, reference_context, results, qa_pairs, retriever_method, retriever_evaluation, automated_evaluation))

            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    print(f'Generated an exception: {exc}')
                else:
                    print(f'Result: {result}')

    # Write results to a JSON file
    with open(f"json_result_files/{config_name[len('config_files/'):-len('_config.ini')]}/results_{config_name[len('config_files/'):-len('.ini')]}_{'queries' if isinstance(query, list) else 'query'}.json", "w") as f:
        json.dump(results, f, indent=4)


    