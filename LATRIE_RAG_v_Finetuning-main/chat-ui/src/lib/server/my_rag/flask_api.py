from flask import Flask, request, jsonify, make_response
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import HTMLDirectoryReader, my_reranking, XMLDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
import faiss
import json
import os
import shutil
from pathlib import Path
#from flask_cors import CORS

from werkzeug.utils import secure_filename

app = Flask(__name__)
#CORS(app)

# model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
# model = AutoModelForCausalLM.from_pretrained(model_name,
#                                              device_map="auto",
#                                              trust_remote_code=False,
#                                              revision="main")
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
    return response

@app.after_request
def after_request(response):
    return add_cors_headers(response)

@app.route('/search', methods=['POST'])
def search():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    document_objects = []

    directory_path = "./my_files"

    xml_dir = os.path.join(directory_path, "XML")
    html_dir = os.path.join(directory_path, "HTML")
    pdf_dir = os.path.join(directory_path, "PDF")

    os.makedirs(xml_dir, exist_ok=True)
    os.makedirs(html_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        
        if os.path.isfile(file_path):
            file_extension = Path(file).suffix.lower()

            if file_extension == '.xml':
                shutil.copy(file_path, os.path.join(xml_dir, file))
            elif file_extension == '.html':
                shutil.copy(file_path, os.path.join(html_dir, file))
            elif file_extension == '.pdf':
                shutil.copy(file_path, os.path.join(pdf_dir, file))

    if os.listdir(xml_dir):  
        documents = XMLDirectoryReader(xml_dir).load_data()
        document_objects.extend([Document(text=doc, title=f"Document {i+1}") for i, doc in enumerate(documents)])

    if os.listdir(html_dir):  
        documents = HTMLDirectoryReader(html_dir).load_data()
        document_objects.extend([Document(text=doc, title=f"Document {i+1}") for i, doc in enumerate(documents)])

    if os.listdir(pdf_dir): 
        documents = SimpleDirectoryReader(pdf_dir).load_data()
        document_objects.extend(documents)

    shutil.rmtree(xml_dir)
    shutil.rmtree(html_dir)
    shutil.rmtree(pdf_dir)

    text_parser = SentenceSplitter(chunk_size=256, chunk_overlap=25)
    text_chunks = []
    doc_idxs = []

    for doc_idx, doc in enumerate(document_objects):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = document_objects[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    for node in nodes:
        node_embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
        node.embedding = node_embedding

    d = len(nodes[0].embedding)
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    vector_store.add(nodes)

    data = request.json
    query = data['query']
    top_k = data.get('top_k', 5)

    query_embedding = embed_model.get_query_embedding(query)
    vector_store_query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=top_k)
    query_result = vector_store.query(vector_store_query)

    context = "Context:\n"
    chunks = []
    for i in range(top_k):
        chunk_text = nodes[int(query_result.ids[top_k-i-1])].text
        chunks.append(chunk_text)
        context += f"{chunk_text}\n\n"

    top_n = data.get('top_n', 2)
    new_chunks, scores = my_reranking("cross-encoder/ms-marco-MiniLM-L-6-v2", query, chunks, top_n)

    response_context = "Context:\n"
    for i in range(top_n):
        response_context += f"{new_chunks[i]}\n\n"

    return jsonify({
        "original_context": context,
        "original_chunks": chunks,
        "reranked_context": response_context,
        "new_chunks": new_chunks
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save('my_files/' + filename)
    response = jsonify({'message': f'Fichier {filename} envoyé avec succès!'})
    return response

@app.route('/delete-file', methods=['DELETE'])
def delete_file():
    file_name = request.args.get('fileName')
    try:
        os.remove(os.path.join("my_files/", file_name))
        response = jsonify({'message': f'Fichier {file_name} supprimé avec succès!'})
        return response
    except FileNotFoundError:
        response = jsonify({'message': f'Fichier {file_name} non trouvé'}), 404
        return response
    except Exception as e:
        response = jsonify({'message': f'Erreur lors de la suppression du fichier {file_name}'}), 500
        return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    #app.run(host='127.0.0.1', port=8080)
