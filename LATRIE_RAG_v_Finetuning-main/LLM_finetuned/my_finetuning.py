import sys
sys.path.insert(0, '../')
import configparser
import itertools
from utils import get_choice, HTMLDirectoryReader, get_answer_by_question, is_unique_list_of_ints, mean_max_tokens, get_max_new_tokens
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
import time

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


def inference(model, query, qa_pairs, llm, tokenizer, device):
    model.eval()

    if isinstance(query, list):
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

        if llm == "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ" or llm == "mistralai/Mistral-7B-Instruct-v0.2":
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

    else:
        # prompt (no additional prompt)
        prompt1 = f'''[INST] {query} [/INST]'''

        messages1 = [
            {"role": "user", "content": query}
        ]

        # prompt (additional promp)
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


        prompts = {
            1: prompt1,
            2: prompt2
        }

        messages = {
            1: messages1,
            2: messages2
        }

        responses = {}

        max_new_tokens = get_max_new_tokens(llm, qa_pairs[query-1]['answer']) 
        if llm == "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ" or llm == "mistralai/Mistral-7B-Instruct-v0.2":
            for i in range(1, 3):
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

                responses[i] = response
        else:
            pipe = pipeline( 
                "text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                device=device.index if device.type == 'cuda' else -1
            ) 

            generation_args = { 
                "max_new_tokens": max_new_tokens +50, 
                "return_full_text": False, 
                "temperature": 0.0, 
                "do_sample": False, 
            } 

            for i in range(1, 3):
                output = pipe(messages[i], **generation_args) 
                response = output[0]['generated_text']

                responses[i] = response

    return responses


def results_for_many_queries(candidates, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    metrics_score = {}

    for j in range(1, 3):
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


def results_for_one_query(candidates, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    metrics_score = {}

    for i in range(1, 3):
        metrics_score[i] = {}

        bleu_score = {}
        bleu_score["1_gram"] = sentence_bleu(reference.split(), candidates[i].answer, weights=(1, 0, 0, 0))
        bleu_score["2_gram"] = sentence_bleu(reference.split(), candidates[i].answer, weights=(0, 1, 0, 0))
        bleu_score["3_gram"] = sentence_bleu(reference.split(), candidates[i].answer, weights=(0, 0, 1, 0))
        bleu_score["4_gram"] = sentence_bleu(reference.split(), candidates[i].answer, weights=(0, 0, 0, 1))
    
        rouge_score = {}
        rouge_scores = scorer.score(reference, candidates[i].answer)
        rouge_score["rouge_1"] = {}
        rouge_score["rouge_1"]["precision"] = rouge_scores['rouge1'].precision
        rouge_score["rouge_1"]["recall"] = rouge_scores['rouge1'].recall
        rouge_score["rouge_1"]["fmeasure"] = rouge_scores['rouge1'].fmeasure
        rouge_score["rouge_L"] = {}
        rouge_score["rouge_L"]["precision"] = rouge_scores['rougeL'].precision
        rouge_score["rouge_L"]["recall"] = rouge_scores['rougeL'].recall
        rouge_score["rouge_L"]["fmeasure"] = rouge_scores['rougeL'].fmeasure

        m_score = meteor_score([reference.split()], candidates[i].answer.split())
        
        bertscore = {}
        P, R, F1 = score([candidates[i].answer], [reference], lang='en', rescale_with_baseline=True)
        bertscore["Precision"] = round(P.mean().item(), 3)
        bertscore["Recall"] = round(R.mean().item(), 3)
        bertscore["F1_measure"] = round(F1.mean().item(), 3)

        metrics_score[i]["BLEU"] = bleu_score
        metrics_score[i]["ROUGE"] = rouge_score
        metrics_score[i]["METEOR"] = m_score
        metrics_score[i]["BERTScore"] = bertscore

    return metrics_score


def finetune_model(base_model, tokenizer, tokenized_train_data, tokenized_test_data, rank, lora_alpha, lora_dropout, lr, batch_size, num_epochs, target_modules, quantization):
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
    
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    training_time = end_time - start_time

    # renable warnings
    peft_model.config.use_cache = True

    return peft_model, training_time


def evaluate_model(llm, rank, lora_alpha, lora_dropout, lr, batch_size, num_epochs, query, reference_answer, results, qa_pairs, target_modules, quantization):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if quantization and llm != "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ":
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
    else:
        base_model = AutoModelForCausalLM.from_pretrained(llm,
                                                    device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
                                                    trust_remote_code=False, # prevents running custom model files on your machine
                                                    revision="main") # which version of model to use in repo

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
    if isinstance(query, list):
        base_metrics = results_for_many_queries(base_answer, reference_answer)

    else:
        base_metrics = results_for_one_query(base_answer, reference_answer)

    peft_model, training_time = finetune_model(base_model, tokenizer, tokenized_train_data, tokenized_test_data, rank, lora_alpha, lora_dropout, lr, batch_size, num_epochs, target_modules, quantization)

    finetuned_answer = inference(peft_model, query, qa_pairs, llm, tokenizer, device)
    if isinstance(query, list):
        finetuned_metrics = results_for_many_queries(finetuned_answer, reference_answer)

    else:
        finetuned_metrics = results_for_one_query(finetuned_answer, reference_answer)

    if isinstance(query, list):
        queries_entry = {
            "queries": [qa_pairs[i-1]['question'] for i in query],
            "reference answers": reference_answer,
        }
    else:
        queries_entry = {
            "query": query,
            "reference answer": reference_answer,
        }

    entry = OrderedDict({
        "parameters": {
            "llm": llm,
            "rank": rank,
            "lora_alpha": lora_alpha,
            "target modules": target_modules,
            "lora_dropout": lora_dropout,
            "quantization": quantization and llm != "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        },
        "training time": training_time,
        **queries_entry,  
        "predicted answers": {
            "no additional prompt / no finetuning": base_answer[1],
            "additional prompt / no finetuning": base_answer[2],
            "no additional prompt / finetuning": finetuned_answer[1],
            "additional prompt / finetuning": finetuned_answer[2],
        },
        "metrics": {
            "no additional prompt / no finetuning": base_metrics[1],
            "additional prompt / no finetuning": base_metrics[2],
            "no additional prompt / finetuning": finetuned_metrics[1],
            "additional prompt / finetuning": finetuned_metrics[2],
        }
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

    #query = get_choice('query', options_query)
    # if is_unique_list_of_ints(query):
    #     query = ast.literal_eval(query)
    #query = [9, 12, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165]
    #query = [9, 12, 32, 43, 44, 52, 54, 61, 66, 70, 78, 85, 87, 89, 90, 93, 97, 101, 104, 109, 111, 117, 118, 123, 125, 126, 132, 138, 140, 142, 147, 150, 152, 153, 159, 161, 164]
    query = [9, 12, 46, 76, 93, 123, 159]
    #query = list(range(1, 166))
    if is_unique_list_of_ints(query):
        query = ast.literal_eval(query)
    
    if isinstance(query, list):
        reference_answer = [qa_pairs[i-1]['answer'] for i in query]
    else:
        reference_answer = get_answer_by_question(query, qa_pairs)


    #config_name = get_choice('config', options_config)
    config_name = "config_files/target_modules_config.ini"

    config = configparser.ConfigParser()

    config.sections()

    config.read(config_name)

    # Get params from config 
    llm = eval(config.get('models', 'LLM'))
    rank = eval(config.get('lora', 'rank'))
    lora_alpha = eval(config.get('lora', 'lora_alpha'))
    target_modules = eval(config.get('lora', 'target_modules'))
    lora_dropout = eval(config.get('lora', 'lora_dropout'))
    lr = eval(config.get('train', 'lr'))
    batch_size = eval(config.get('train', 'batch_size'))
    num_epochs = eval(config.get('train', 'num_epochs'))
    quantization = eval(config.get('quantization', 'used'))

    paramlist = [llm, rank, lora_alpha, lora_dropout, lr, batch_size, num_epochs, target_modules, quantization]

    results = []

    for test in itertools.product(*paramlist):
        llm = test[0]
        rank = test[1]
        lora_alpha = test[2]
        lora_dropout = test[3]
        lr = test[4]
        batch_size = test[5]
        num_epochs = test[6]
        target_modules = test[7]
        quantization = test[8]
        evaluate_model(llm, rank, lora_alpha, lora_dropout, lr, batch_size, num_epochs, query, reference_answer, results, qa_pairs, target_modules, quantization)

    # Write results to a JSON file
    output_dir = f"json_result_files/{config_name[len('config_files/'):-len('_config.ini')]}"
    output_file = f"{output_dir}/results_{config_name[len('config_files/'):-len('.ini')]}_{'queries' if isinstance(query, list) else 'query'}.json"

    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    

