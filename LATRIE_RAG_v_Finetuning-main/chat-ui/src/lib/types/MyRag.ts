import type { ObjectId } from "mongodb";
import type { Conversation } from "./Conversation";
import type { Timestamps } from "./Timestamps";

export interface MyRag extends Timestamps {
	_id?: ObjectId;
	convId?: Conversation["_id"];

	prompt: string;

	original_context: string;

    original_chunks: string[];

    reranked_context: string;

    new_chunks: string[];
}