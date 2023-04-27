import asyncio
import itertools
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt
import openai
import tiktoken

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_BATCH_SIZE = 500


@dataclass
class Document:
    text: str
    vector: npt.NDArray[np.float32]


class VectorDB:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        # Yes the vector db is a np.ndarray. Using Pinecone/ANN would be better for ##MUTCH## larger datasets.
        self.vectors = np.stack([doc.vector for doc in documents])

    def search(self, vector: npt.NDArray[np.float32], k: int = 5, min_similarity=0.0) -> list[Document]:
        """Search for k nearest documents to the given vector with cosine similarity"""
        scores = np.dot(self.vectors, vector) / (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(vector))
        indices = np.argpartition(scores, -k)[-k:]
        return [self.documents[i].text for i in indices if scores[i] > min_similarity]


def chunks(document: str, max_len: int, tokenizer):
    """create chunks of texts that contain full sentences and are not longer than max_len.
    In case a sentence is longer than max_len, it will NOT be split into multiple chunks.
    """

    doc_t = tokenizer.encode(document)
    dot_t = tokenizer.encode(".")[0]
    dot2_t = tokenizer.encode(" . ")[0]
    end_indices = [i for i, c in enumerate(doc_t) if c in [dot_t, dot2_t]]
    current_start = 0
    current_end = end_indices[0]

    for i in end_indices[1:]:
        if i - current_start <= max_len:
            current_end = i
        else:
            yield tokenizer.decode(doc_t[current_start : current_end + 1])
            current_start = current_end + 1
            current_end = i
    if current_end - current_start > 0:
        yield tokenizer.decode(doc_t[current_start : current_end + 1])


def batch_generator(iterable: Iterable, batch_size: int):
    """Yield successive batch_size chunks from the iterable."""
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch


async def acreate_docs_batch(text_batch: list[str], openai_key: str) -> list[Document]:
    batch_response = await openai.Embedding.acreate(model=EMBEDDING_MODEL, input=text_batch, api_key=openai_key)
    batch_embeddings = [np.array(e["embedding"]) for e in batch_response["data"]]
    return [Document(text, embedding) for text, embedding in zip(text_batch, batch_embeddings)]


async def index_long_document(large_document: str, openai_key: str, chunk_size: int = 300) -> VectorDB:
    tokenizer = tiktoken.get_encoding("gpt2")
    docs = await asyncio.gather(
        *[
            acreate_docs_batch(text_batch=text_batch, openai_key=openai_key)
            for text_batch in batch_generator(chunks(large_document, chunk_size, tokenizer), EMBEDDING_BATCH_SIZE)
        ]
    )
    docs = list(itertools.chain(*docs))  # flatten
    db = VectorDB(docs)
    return db


async def topk_chunks(question: str, db: VectorDB, openai_key: str, k: int = 5, min_similarity=0.0) -> list[str]:
    response = await openai.Embedding.acreate(model=EMBEDDING_MODEL, input=[question], api_key=openai_key)
    question_embedding = np.array(response["data"][0]["embedding"])
    return db.search(question_embedding, k=k, min_similarity=min_similarity)
