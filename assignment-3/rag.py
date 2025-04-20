import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

documents = [
    "The Eiffel Tower is located in Paris.",
    "Python is a popular programming language.",
    "The moon is Earth's only natural satellite.",
]

# Encode documents
encoder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = encoder.encode(documents, convert_to_numpy=True)

# Create FAISS index - we should have different versions here
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)


# Retrieval function
def retrieve(query, top_k=1):
    query_vec = encoder.encode([query])
    distances, indices = index.search(np.array(query_vec), k=top_k)
    return [documents[i] for i in indices[0]]


# Generation
lm_path = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(lm_path)
generator = AutoModelForSeq2SeqLM.from_pretrained(lm_path)


def generate_answer(query, retrieved_docs):
    context = " ".join(retrieved_docs)
    prompt = f"Context: {context} Question: {query} Answer:"
    print(f"Prompting LM: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = generator.generate(**inputs, max_new_tokens=50)
    # print(f"Outputs: {outputs}")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    query = "Where is Eiffel Tower located in?"
    retrieved_docs = retrieve(query, top_k=1)
    answer = generate_answer(query, [])

    print(answer)
