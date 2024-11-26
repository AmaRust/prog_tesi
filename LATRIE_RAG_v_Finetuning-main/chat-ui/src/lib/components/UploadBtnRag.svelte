<script lang="ts">
	import CarbonUpload from "~icons/carbon/upload";
  
	export let classNames = "";
	export let files: File[];
	export let mimeTypes: string[];
  
	/**
	 * Due to a bug with Svelte, we cannot use bind:files with multiple
	 * So we use this workaround
	 **/
	const onFileChange = async (e: Event) => {
	  	if (!e.target) return;
	  	const target = e.target as HTMLInputElement;
	  	files = [...files,...(target.files?? [])];
  
	  	if (target.files) {
			for (const file of target.files) {
				const formData = new FormData();
				formData.append("file", file);
		
				try {
					const response = await fetch("http://127.0.0.1:8080/upload", {
						method: "POST",
						body: formData,
					});
		
					if (response.ok) {
						console.log("Fichier envoyé avec succès!");
					} else {
						console.error("Erreur lors de l'envoi du fichier");
					}
				} catch (error) {
					console.error("Erreur lors de l'envoi du fichier", error);
				}
			};
	  	}
	}
  </script>
  
  <button
	class="btn relative h-8 rounded-lg border bg-white px-3 py-1 text-sm text-gray-500 shadow-sm hover:bg-gray-100 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600 {classNames}"
  >
	<input
	  	class="absolute w-full cursor-pointer opacity-0"
	  	type="file"
	  	on:change={onFileChange}
	  	accept={mimeTypes.join(",")}
	/>
	<CarbonUpload class="mr-2 text-xxs" /> Upload file
  </button>



