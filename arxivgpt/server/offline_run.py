import asyncio
import os

from main import get_pdf
from question_answering import refine_qa
from vector import index_long_document, topk_chunks

OPENAI_KEY = os.environ["OPENAI_KEY"]

if __name__ == "__main__":
    pdf = asyncio.run(get_pdf("https://arxiv.org/abs/2010.11929"))
    large_document = " ".join(p.extract_text() for p in pdf.pages)
    db = asyncio.run(index_long_document(large_document, OPENAI_KEY, chunk_size=500))

    question = "On what dataset was the model trained?"
    result = asyncio.run(topk_chunks(question, db, OPENAI_KEY))
    answer = asyncio.run(refine_qa(question, result, OPENAI_KEY))
    print(answer)
