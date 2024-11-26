import { writable } from "svelte/store";
export interface MyRagParameters {
	useRag: boolean;
	nItems: number;
}
export const myRagParameters = writable<MyRagParameters>({
	useRag: false,
	nItems: 5,
});
