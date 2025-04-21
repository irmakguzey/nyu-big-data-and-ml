import time

import faiss
import numpy as np
import torch.utils.data as data
from data_utils import get_datasets
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

np.set_printoptions(precision=3, suppress=True)


# This class will input param for using RAG or not
class AnswerGenerator:
    def __init__(self, lm_path):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_path)
        self.generator = AutoModelForCausalLM.from_pretrained(lm_path)

        # Convert every chunk in our dataset into the encoded embeddings
        self.max_length = 256
        train_dataset, test_dataset, _ = get_datasets(
            root_dir="climate_text_dataset",
            preprocess=False,
            model_name=lm_path,
            max_length=self.max_length,
        )
        self.train_loader = data.DataLoader(
            train_dataset,
            batch_size=64,
            num_workers=4,
        )
        self.test_loader = data.DataLoader(
            test_dataset,
            batch_size=64,
            num_workers=4,
        )
        # if use_rag:
        #     self.use_rag = True
        # self.set_encoder(encoder_path)
        # self._set_doc_embeddings()
        # self.set_index(rag_type)

    def set_use_rag(self, use_rag):
        self.use_rag = use_rag

    def set_encoder(self, encoder_path):
        if self.use_rag:
            self.encoder = SentenceTransformer(encoder_path)
            self._set_doc_embeddings()
        else:
            print("set_encoder - Not using RAG passing")

    def set_index(self, rag_type):
        if self.use_rag:
            if rag_type == "index_flat_l2":
                self.index = faiss.IndexFlatL2(self.doc_embeddings.shape[1])
                self.index.add(self.doc_embeddings)
            elif rag_type == "index_hnsw":
                self.index = faiss.IndexHNSWFlat(self.doc_embeddings.shape[1], 32)
                self.index.add(self.doc_embeddings)
            elif rag_type == "index_ivf":
                N, d = self.doc_embeddings.shape
                # nlist = int(np.sqrt(N))  # ~100
                nlist = 15
                quantizer = faiss.IndexFlatL2(d)
                self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
                self.index.train(self.doc_embeddings)
                self.index.add(self.doc_embeddings)
                self.index.nprobe = 16
            else:
                raise ValueError(f"Invalid RAG type: {rag_type}")
        else:
            print("set_index - Not using RAG passing")

    def _set_doc_embeddings(self):
        # Convert all documents from train loader into embeddings
        doc_embeddings = []
        doc_texts = []
        for batch in self.train_loader:
            # Encode each batch of documents
            batch_embeddings = self.encoder.encode(batch["text"], convert_to_numpy=True)
            doc_embeddings.append(batch_embeddings)
            doc_texts.append(batch["text"])

        # Concatenate all batch embeddings into single array
        self.doc_embeddings = np.concatenate(doc_embeddings, axis=0)
        self.doc_texts = np.concatenate(doc_texts, axis=0)

        print(
            f"Document embeddings shape: {self.doc_embeddings.shape} - doc texts: {self.doc_texts.shape}"
        )

    def generate_answer(self, query, top_k=1):
        if self.use_rag:  # Add context to the prompt
            # Retrieve docs
            query_vec = self.encoder.encode([query])
            distances, indices = self.index.search(np.array(query_vec), k=top_k)
            retrieved_docs = [self.doc_texts[i] for i in indices[0]]

            # Generate answer
            context = " ".join(retrieved_docs)
            prompt = f"Context: {context} Question: {query} Answer:"

        else:
            # Generate answer using only LM
            prompt = f"Question: {query} Answer:"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=self.max_length, truncation=True
        )
        outputs = self.generator.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    finetuned_llama_path = "/scratch/ig2283/Workspace/nyu-big-data-and-ml/assignment-1/results-llamba-03-11_18:16:23/checkpoint-200"
    pretrained_llama_path = (
        "/scratch/ig2283/Workspace/nyu-big-data-and-ml/assignment-1/Llama3.2-3B"
    )

    # llama_path_dict = {
    #     "finetuned": "google/flan-t5-base",
    #     "pretrained": "google/flan-t5-base",
    # }
    llama_path_dict = {
        "finetuned": finetuned_llama_path,
        "pretrained": pretrained_llama_path,
    }

    model_types = ["finetuned", "pretrained"]
    use_rags = [False, True]
    encoder_names = ["all-MiniLM-L6-v2", "BAAI/bge-large-en"]
    rag_types = ["index_hnsw", "index_ivf", "index_flat_l2"]

    # Open log file in append mode
    # query = "How does CO2 affect the climate?"
    timestamp = time.strftime("%m%d-%H%M%S")
    log_filename = f"log_{timestamp}.txt"

    queries = [
        "What is a method to measure CO2 leakage from carbon capture and sequestration?",
        "How are Class Activation Mapping methods used to improve CO2 leakage detection from carbon capture and sequestration?",
        "How can machine learning be used to predict vegetation health and mitigate the impacts of agricultural drought in Kenya?",
        "How can machine learning practitioners estimate the energy consumption of their models without training them?",
        "How can machine learning aid in detecting CO2 leakage in carbon capture and sequestration (CCS) projects?",
    ]

    for query in queries:
        for model_type in model_types:
            print(f"Model type: {model_type} - Path: {llama_path_dict[model_type]}")
            gen = AnswerGenerator(
                lm_path=llama_path_dict[model_type],
            )
            for use_rag in use_rags:
                gen.set_use_rag(use_rag)
                if use_rag:
                    for encoder_name in encoder_names:
                        gen.set_encoder(encoder_name)
                        for rag_type in rag_types:
                            gen.set_index(rag_type)

                            time_start = time.time()
                            answer = gen.generate_answer(query, top_k=10)
                            time_end = time.time()
                            time_spent = time_end - time_start
                            print(answer)
                            print(
                                f"RAG type: {rag_type} - Time taken: {time_spent:.3f} seconds"
                            )
                            with open(log_filename, "a") as f:
                                f.write(f"Model type: {model_type}\n")
                                f.write(f"Use RAG: {use_rag}\n")
                                f.write(f"Encoder: {encoder_name}\n")
                                f.write(f"RAG type: {rag_type}\n")
                                f.write(f"Time taken: {time_spent:.3f} seconds\n")
                                f.write(f"Query: {query} - Answer: {answer}\n")
                                f.write("-" * 50 + "\n")

                else:

                    time_start = time.time()
                    answer = gen.generate_answer(query, top_k=10)
                    time_end = time.time()
                    time_spent = time_end - time_start
                    print(answer)
                    print(f"Time taken: {time_spent:.3f} seconds")
                    with open(log_filename, "a") as f:
                        f.write(f"Model type: {model_type}\n")
                        f.write(f"Use RAG: {use_rag}\n")
                        f.write(f"Time taken: {time_spent:.3f} seconds\n")
                        f.write(f"Query: {query} - Answer: {answer}\n")
                        f.write("-" * 50 + "\n")

    # gen = AnswerGenerator(
    #     lm_path="google/flan-t5-base",
    #     use_rag=True,
    # )

    # for encoder_name in encoder_names:
    #     gen.set_encoder(encoder_name)
    #     for rag_type in rag_types:
    #         gen.set_index(rag_type)
