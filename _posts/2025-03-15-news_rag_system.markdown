---
layout: post
title:  "A simple RAG+LLM System for News"
date:   2025-03-15 10:33:15 +0200
categories: jekyll update
---

# A simple RAG+LLM System for News

Given the current hype about LLM and RAGs I build a small protoype for a RAG+LLM system for the [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) from Kaggle. The dataset contains 210k news headlines with short descritions and meta data from 2012 to 2022 from HuffPost.
The dataset is a collection of JSON objects for each news entry.
Code can be found [here]()

## RAG

<div style="text-align: center">
    <img src="{{ '/assets/img/rag.png' | relative_url }}" alt="RAG-sketch" title="RAG" width="500"/>
</div>  

Retrieval-Augmented Generation (RAG) is a framework that enhances the performance of large language models (LLMs) by combining them with external knowledge sources. Traditional LLMs, such as GPT or BERT-based models, generate text based on patterns learned during training, but they are limited by their fixed knowledge cutoff and can sometimes hallucinate incorrect or outdated information. RAG addresses this by retrieving relevant documents from an external corpus (like a database or document store) and incorporating that information into the generation process.  

The RAG pipeline typically involves two main components: a retriever and a generator. The retriever selects the most relevant documents based on a query. These documents are then passed to the generator, usually an LLM, which uses them as context to produce a more accurate and grounded answer.  

Since I am doing this on my local machine I choose to use [*all-MiniLM-L6-v2*](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for the embedding model and [*GPT2*](https://huggingface.co/openai-community/gpt2) for the LLM. Also I just used a maximum of *128* tokens for the generator and just retrieve the top *k=3* results from the RAG.

## Loading Data
The prepare_data function processes a JSON file containing news data by loading and parsing each line to extract entries with both a headline and short_description. It validates the file's existence, reads the entries, and skips invalid JSON lines with a warning.   
The function returns three outputs: a list of dictionaries with the full data, a list of headlines, and a list of short descriptions, making it useful for preprocessing tasks like text analysis or machine learning.

```python
import json
import os

def prepare_data(json_path: str):
    [...docstrings & error handling...]

    documents = []
    with open(json_path, "r") as f:
        for line in f:
            try:
                # Parse each line as a JSON object
                item = json.loads(line.strip())
                if item.get("headline") and item.get("short_description"):
                    documents.append(item)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")

    headlines = [item["headline"] for item in documents]
    descriptions = [item["short_description"] for item in documents]

    [...]
```

### Retriever
I generate embeddings for a list of text descriptions using a specified *SentenceTransformer* model and build a *FAISS* index for efficient similarity search. It first validates that the input is a list of strings, then uses the embedding model to encode the descriptions into normalized embeddings.  
The dimensionality of the embeddings is determined, and a FAISS IndexFlatIP (inner product-based index) is created and populated with the embeddings, enabling fast retrieval of similar items.

```python
def prepare_embeddings_and_index(self, descriptions: list, embedding_model: str):
        [...docstrings & error handling...]

        self.embedder = SentenceTransformer(embedding_model)
        doc_embeddings = self.embedder.encode(descriptions, normalize_embeddings=True)
        dimension = doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(doc_embeddings)
```

## RAG Query and LLM
The rag_query method performs a retrieval-augmented generation (RAG) process to answer a question using provided headlines and descriptions. It encodes the question into an embedding, retrieves the top k most relevant descriptions and their corresponding headlines using a FAISS index, and constructs a prompt combining the retrieved contexts and the question.  
This prompt is passed to a text generation model to produce a response, from which the final answer is extracted. The method returns the generated answer and the retrieved contexts.

```python
def rag_query(
        self,
        question: str,
        headlines: list[str],
        descriptions: list[str],
        top_k: int = 3,
        max_tokens: int = 128,
        verbose:bool = True,
    ) -> Tuple[str, List[str]]:
        [...docstrings & error handling...]
        
        q_embedding = self.embedder.encode([question], normalize_embeddings=True)
        scores, indices = self.index.search(q_embedding, top_k)

        retrieved_contexts = [descriptions[i] for i in indices[0]]
        retrieved_headlines = [headlines[i] for i in indices[0]]

        # Compose prompt
        context_block = "\n".join(f"- {ctx}" for ctx in retrieved_contexts)
        prompt = f"""Use the following information to answer the question.

        Context:
    {context_block}

    Question: {question}
    """

        print(prompt)

        # Generate response
        response = self.generator(prompt, max_new_tokens=max_tokens, do_sample=False)
        answer = response[0]["generated_text"].split("Answer:")[-1].strip()

        [...]
```

