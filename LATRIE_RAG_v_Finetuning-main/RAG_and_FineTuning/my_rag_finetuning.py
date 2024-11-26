import sys
sys.path.insert(0, '../')
import configparser
import itertools
from utils import get_choice, HTMLDirectoryReader, get_answer_by_question, is_unique_list_of_ints, mean_max_tokens, get_max_new_tokens, split_into_sentences, my_reranking, get_config_hash, load_embeddings, save_embeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from peft import PeftModel
from datasets import load_dataset
import transformers
from sklearn.model_selection import train_test_split
import json
import torch

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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import Document

def segment_text(data):
    segments = []
    current_segment = []

    for text in data:
        lines = text.split("\n")
        for line in lines:
            if line.startswith("title :") or line.startswith("head :"):
                if current_segment:
                    segments.append("\n".join(current_segment).strip())
                    current_segment = []
            current_segment.append(line)

        if current_segment:
            segments.append("\n".join(current_segment).strip())

    return segments

def tokenize_function(examples, tokenizer):
    # extract text
    text = examples

    # tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    # Convertir les array en listes
    tokenized_inputs = {key: value[0].tolist() for key, value in tokenized_inputs.items()}

    return tokenized_inputs

def splitting_and_context(embedding_model_name, chunk_size, chunk_overlap, document_objects, top_k, query, qa_pairs, reranking, top_n):
    new_context = ""
    new_chunks = []
    new_ids = []

    config_hash = get_config_hash(chunk_size, chunk_overlap, "From scratch with Faiss vector store and SYNTACTIC splitter", embedding_model_name)
    embeddings = load_embeddings(config_hash)

    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)

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
            
    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = document_objects[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

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

    contexts = []
    new_contexts = []

    query_mode = "default"
    # query_mode = "sparse"
    # query_mode = "hybrid"

    for i in query:
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

        if reranking:
            new_context = "Context:\n"
            new_chunks, scores = my_reranking("BAAI/bge-reranker-base", qa_pairs[i-1]['question'], chunks, top_n, []) 
            for i in range(top_n):
                new_context += f"{new_chunks[i]}\n\n"
                        
            new_contexts.append(new_context)

    return contexts, new_contexts

    


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


def inference(model, query, qa_pairs, llm, tokenizer, device):
    model.eval()

    prompts1 = []
    prompts2 = []

    messages1 = []
    messages2 = []

    for i, q in enumerate(query):
        # prompt (no additional prompt)
        prompt1 = f'''[INST] {qa_pairs[q-1]['question']} [/INST]'''
        prompts1.append(prompt1)

        message1 = [
            {"role": "user", "content": qa_pairs[q-1]['question']}
        ]
        messages1.append(message1)

        # prompt (additional prompt)
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


        prompts = {
            1: prompts1,
            2: prompts2,
        }

        messages = {
            1: messages1,
            2: messages2,
        }

    responses = {}

    if llm == "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ":
        for i in range(1, 3):
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
                    #length_penalty=2,
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
        
        for i in range(1, 3):
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

def inference_w_RAG(query, contexts, new_contexts, tokenizer, model, llm, qa_pairs, device):
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


def results_for_many_queries(candidates, references, rag):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    metrics_score = {}

    for j in range(1, 5 if rag else 3):
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


def finetune_model(base_model, tokenizer, tokenized_train_data, tokenized_test_data, rank, lora_alpha, target_modules, lora_dropout, lr, batch_size, num_epochs):
    base_model.train() # model in training mode (dropout modules are activated)

    # enable gradient check pointing
    base_model.gradient_checkpointing_enable()

    # enable quantized training
    base_model = prepare_model_for_kbit_training(base_model)

    # LoRA config
    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # LoRA trainable version of model
    peft_model = get_peft_model(base_model, config)

    # trainable parameter count
    peft_model.print_trainable_parameters()


    # setting pad token
    tokenizer.pad_token = tokenizer.eos_token
    # data collator
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Fine-tuning Model

    # hyperparameters
    lr = lr
    batch_size = batch_size
    num_epochs = num_epochs

    # define training arguments
    training_args = transformers.TrainingArguments(
        output_dir= "Finetuned_Med_GPT",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        fp16=True,
        optim="paged_adamw_8bit",

    )

    # configure trainer
    trainer = transformers.Trainer(
        model=peft_model,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_test_data,
        args=training_args,
        data_collator=data_collator
    )

    # train model
    peft_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    # renable warnings
    peft_model.config.use_cache = True

    return peft_model


def evaluate_model(llm, rank, lora_alpha, target_modules, lora_dropout, lr, batch_size, num_epochs, query, reference_answer, results, qa_pairs, embedding, chunksize, overlap, top_k, reranking, top_n, is_automated_evaluation):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(llm,
                                                device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
                                                trust_remote_code=False, # prevents running custom model files on your machine
                                                revision="main", # which version of model to use in repo
                                                quantization_config=bnb_config) 

    base_model = base_model.to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm, use_fast=True)

    # extract data from HTML files
    data = HTMLDirectoryReader("../data/Medical_Guidelines/HTML").load_data()

    segments = segment_text(data)

    train_segments, val_segments = train_test_split(segments, test_size=0.1, random_state=42)

    tokenized_train_data = [tokenize_function(segment, tokenizer) for segment in train_segments]
    tokenized_test_data = [tokenize_function(segment, tokenizer) for segment in val_segments]

    base_answer = inference(base_model, query, qa_pairs, llm, tokenizer, device)
    rag = 0
    base_metrics = results_for_many_queries(base_answer, reference_answer, rag)

    peft_model = finetune_model(base_model, tokenizer, tokenized_train_data, tokenized_test_data, rank, lora_alpha, target_modules, lora_dropout, lr, batch_size, num_epochs)

    document_objects = [Document(text=doc, title=f"Document {i+1}") for i, doc in enumerate(data)]
    contexts, new_contexts = splitting_and_context(embedding, chunksize, overlap, document_objects, top_k, query, qa_pairs, reranking, top_n)

    finetuned_rag_answer = inference_w_RAG(query, contexts, new_contexts, tokenizer, peft_model, llm, qa_pairs, device)
    rag = 1
    finetuned_metrics = results_for_many_queries(finetuned_rag_answer, reference_answer, rag)

    if is_automated_evaluation:
        automated_evaluation = prompt_evaluation_queries(query, qa_pairs, finetuned_rag_answer[3], contexts, new_contexts)

    entry = OrderedDict({
        "parameters": {
            "llm": llm,
            "Fine-Tuning": {
                "rank": rank,
                "lora_alpha": lora_alpha,
                "target modules": target_modules,
                "lora_dropout": lora_dropout,
                "learning_rate": lr,
                "batch_size": batch_size,
                "num_epochs": num_epochs
            },
            "RAG": {
                "embedding model": embedding,
                "splitting method": "From scratch with Faiss vector store and SYNTACTIC splitter",
                "chunk size": chunksize,
                "overlap": overlap,
                "top_k": top_k,
                "re-ranking": reranking,
                "top_n": top_n
            }
        },
        "queries": [qa_pairs[i-1]['question'] for i in query],
        "reference answers": reference_answer,  
        "predicted answers": {
            "no additional prompt / no finetuning / no rag": base_answer[1],
            "additional prompt / no finetuning / no rag": base_answer[2],
            "no additional prompt / finetuning / no rag": finetuned_rag_answer[1],
            "additional prompt / finetuning / no rag": finetuned_rag_answer[2],
            "no additional prompt / finetuning / rag": finetuned_rag_answer[3],
            "additional prompt / finetuning / rag": finetuned_rag_answer[4],
        },
        "metrics": {
            "no additional prompt / no finetuning / no rag": base_metrics[1],
            "additional prompt / no finetuning / no rag": base_metrics[2],
            "no additional prompt / finetuning / no rag": finetuned_metrics[1],
            "additional prompt / finetuning / no rag": finetuned_metrics[2],
            "no additional prompt / finetuning / rag": finetuned_metrics[3],
            "additional prompt / finetuning / rag": finetuned_metrics[4],
        },
        "automated evaluation": automated_evaluation
    })

    results.append(entry)



if __name__ == "__main__":
    # load the JSON evaluation file
    with open('../data/assessment_set/dataset.json', 'r') as dataset:
        qa_pairs = json.load(dataset)
    set_size = len(qa_pairs)

    options_query = [qa_pair['question'] for qa_pair in qa_pairs]
    
    config_directory_path = "config_files/"
    options_config = glob.glob(config_directory_path + "*.ini")

    #query = [9, 12, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165]
    #query = [9, 12, 32, 43, 44, 52, 54, 61, 66, 70, 78, 85, 87, 89, 90, 93, 97, 101, 104, 109, 111, 117, 118, 123, 125, 126, 132, 138, 140, 142, 147, 150, 152, 153, 159, 161, 164]
    query = [9, 12, 46, 76, 93, 123, 159]
    #query = list(range(1, 166))
    if is_unique_list_of_ints(query):
        query = ast.literal_eval(query)
    
    reference_answer = [qa_pairs[i-1]['answer'] for i in query]
    
    results = []

    #Fine-Tuning
    llm = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    rank = 8
    lora_alpha = 64
    target_modules = ["q_proj"]
    lora_dropout = 0.05
    lr = 0.0002
    batch_size = 10
    num_epochs = 200

    #RAG
    embedding = "BAAI/bge-small-en-v1.5"
    chunksize = 256
    overlap = 25
    top_k = 8
    reranking = 1
    top_n = 4
    automated_evaluation = 0

    evaluate_model(llm, rank, lora_alpha, target_modules, lora_dropout, lr, batch_size, num_epochs, query, reference_answer, results, qa_pairs, embedding, chunksize, overlap, top_k, reranking, top_n, automated_evaluation)

    # Write results to a JSON file
    output_dir = f"json_result_files/"
    output_file = f"{output_dir}/results_queries.json"

    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)