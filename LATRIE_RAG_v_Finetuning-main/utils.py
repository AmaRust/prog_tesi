from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from bs4 import BeautifulSoup
import os
import json
import numpy as np
import hashlib
from pydantic import BaseModel, Field
from typing import List
import ast
import re

import nltk
nltk.download('wordnet')
from nltk.translate.meteor_score import meteor_score

def is_unique_list_of_ints(user_input):
    try:
        # Parse the input to a list
        query_list = ast.literal_eval(user_input)
        # Check if it is a list of integers
        if isinstance(query_list, list) and all(isinstance(i, int) for i in query_list):
            # Check for duplicate values
            if len(query_list) == len(set(query_list)):
                return True
        return False
    except (ValueError, SyntaxError):
        return False

def get_choice(variable_name, options):
    print(f"Please choose a value for {variable_name} from the following options:")
    if variable_name == 'query':
        print("0. Choose a set of queries")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")

    while True:
        try:
            choice = int(input(f"Enter the number of your choice for {variable_name}: "))
            if variable_name == 'query' and choice == 0:
                while True:
                    print("1. Enter a list of queries (ex: [2, 4, 67])")
                    print("2. Use all the dataset")
                    sub_choice = int(input("Enter 1 or 2: "))
                    if sub_choice == 1:
                        while True:
                            user_input = input("Enter a list of queries (ex: [2, 4, 67]): ")
                            if is_unique_list_of_ints(user_input):
                                return eval(user_input)
                            else:
                                print("Invalid input. Please enter a list of unique integers.")
                    elif sub_choice == 2:
                        return list(range(1, 166))
                    else:
                        print("Invalid choice. Please enter 1 or 2.")
            elif 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print(f"Invalid choice. Please choose a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid entry. Please enter a number.")


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
    

def get_answer_by_question(question, dataset):
    for entry in dataset:
        if entry["question"] == question:
            return entry["answer"]
    return "Question not found."

def get_context_by_question(question, dataset):
    for entry in dataset:
        if entry["question"] == question:
            return entry["context"]
    return "Question not found."


# reranking method to enhance the relevance of retrieved docs
def my_reranking(model_name, query, passages, top_n, ids): # model_name : "cross-encoder/ms-marco-MiniLM-L-6-v2"
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

    if ids:
        new_ids = [ids[passages.index(passage)] for passage in best_passages]

        return best_passages, scores, new_ids

    return best_passages, scores


def get_config_hash(chunk_size, chunk_overlap, splitting_method, embedding_model):
    config_str = json.dumps({
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "splitting_method": splitting_method,
        "embedding_model": embedding_model
    }, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def save_embeddings(nodes, config_hash, directory="embeddings"):
    os.makedirs(directory, exist_ok=True)
    embeddings_filepath = os.path.join(directory, f"embeddings_{config_hash}.npy")
    #metadata_filepath = os.path.join(directory, f"metadata_{config_hash}.json")
    
    embeddings = np.array([node.embedding for node in nodes])
    np.save(embeddings_filepath, embeddings)
    
    # metadata = [node.metadata for node in nodes]
    # with open(metadata_filepath, "w") as f:
    #     json.dump(metadata, f)

def load_embeddings(config_hash, directory="embeddings"):
    embeddings_filepath = os.path.join(directory, f"embeddings_{config_hash}.npy")
    #metadata_filepath = os.path.join(directory, f"metadata_{config_hash}.json")
    
    if os.path.exists(embeddings_filepath): #and os.path.exists(metadata_filepath):
        embeddings = np.load(embeddings_filepath)
        # with open(metadata_filepath, "r") as f:
        #     metadata = json.load(f)
        return embeddings#, metadata
    return None#, None


class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )

class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )

def count_tokens_transformers(text, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def mean_max_tokens(model_name, qa_pairs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    answers = [qa_pair['answer'] for qa_pair in qa_pairs]

    nb_tokens = [len(tokenizer.tokenize(answers[i])) for i in range(len(answers))]

    return np.ceil(np.mean(nb_tokens))

def get_max_new_tokens(model_name, reference_answer):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return len(tokenizer.tokenize(reference_answer))

def compute_DCG(relevant_scores, k):
    res = 0
    for i in range(1, k+1):
        res += relevant_scores[i-1]/np.log2(i+1)

    return res


def compute_NDCG(relevant_scores, k):
    return compute_DCG(relevant_scores, k)/compute_DCG(sorted(relevant_scores, reverse=True), k)


def compute_MRR(relevant_scores, k):
    if all(x <= 0.5 for x in relevant_scores):
        return 0

    for i in range(k):
        if relevant_scores[i] > 0.5:
            return 1/(i+1)


def compute_MAP(relevant_scores, k):
    N = sum(1 for x in relevant_scores if x > 0.5)

    if N == 0:
        return 0

    res = 0
    for i in range(k):
        if relevant_scores[i] > 0.5:
            res += sum(1 for x in relevant_scores[:i+1] if x > 0.5)/(i+1)

    return res/N


def my_reranking_2(llm, model, tokenizer, device, query, reference_answer, passages, top_n, ids):
    responses = []

    prompt_template_w_context = lambda context, question: f"""[INST]
    {context}
    Please respond to the following question. Use the context above if it is helpful.

    {question}
    [/INST]
    """

    max_new_tokens = get_max_new_tokens(llm, reference_answer)

    for chunk in passages:
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
    
    chunk_to_score = dict(zip(chunk_scores, passages))
    best_scores = sorted(chunk_scores, reverse=True)[:top_n]
    best_passages = [chunk_to_score[score] for score in best_scores]

    if ids:
        new_ids = [ids[passages.index(passage)] for passage in best_passages]

        return best_passages, best_scores, new_ids

    return best_passages, best_scores


def split_into_sentences(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return [sentence.strip() for sentence in sentences]