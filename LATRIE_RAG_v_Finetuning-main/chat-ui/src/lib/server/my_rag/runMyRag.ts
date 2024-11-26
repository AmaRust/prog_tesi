import { defaultEmbeddingModel, embeddingModels } from "$lib/server/embeddingModels";

import type { Conversation } from "$lib/types/Conversation";
import type { Message } from "$lib/types/Message";

import type { MyRag } from "$lib/types/MyRag";

export async function* runMyRag (
	conv: Conversation,
	messages: Message[],
){
	const prompt = messages[messages.length - 1].content;
	const createdAt = new Date();
	const updatedAt = new Date();

	const embeddingModel =
		embeddingModels.find((m) => m.id === conv.embeddingModel) ?? defaultEmbeddingModel;
	if (!embeddingModel) {
		throw Error(`Embedding model ${conv.embeddingModel} not available anymore`);
	}

    const response = await fetch('http://127.0.0.1:8080/search', {  //'http://160.78.28.32:8080/search'
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query: prompt,
			embedding_model: embeddingModel,
            top_k: 6,
            top_n: 3
        })
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
	const original_context = data.original_context;
	const original_chunks = data.original_chunks;
	const reranked_context = data.reranked_context;
	const new_chunks = data.new_chunks;

	const myRag: MyRag = {
		prompt,
		original_context,
		original_chunks,
		reranked_context,
		new_chunks,
		createdAt,
		updatedAt,
	};

	return myRag;
}

