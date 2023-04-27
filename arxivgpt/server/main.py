import io
import os
from urllib.parse import urljoin

import aiohttp
import PyPDF2
from bs4 import BeautifulSoup
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from lru import LRU
from question_answering import refine_qa
from vector import index_long_document, topk_chunks

OPENAI_KEY = os.environ["OPENAI_KEY"]

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# not scalable at all should be replaced with a real database
lru = LRU(maxsize=20)


async def get_pdf(arxiv_link: str) -> PyPDF2.PdfReader:
    async with aiohttp.ClientSession() as session:
        async with session.get(arxiv_link) as resp:
            page = await resp.text()
            soup = BeautifulSoup(page, "html.parser")
            pdf_link = soup.find("a", string="Download PDF")
        if pdf_link:
            pdf_url = urljoin(arxiv_link, pdf_link["href"])
            async with session.get(pdf_url) as resp:
                pdf_bytes = await resp.read()
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                return pdf_reader
        else:
            print(f"PDF link not found. {arxiv_link}")
    return None


async def index_arxiv(arxiv_link: str = Form(...)):
    pdf_reader = await get_pdf(arxiv_link)
    if not pdf_reader:
        return "PDF link not found."
    pdf_text = " ".join(p.extract_text() for p in pdf_reader.pages)
    chunks_db = await index_long_document(pdf_text, OPENAI_KEY, chunk_size=500)
    lru[arxiv_link] = chunks_db
    return chunks_db


@app.post("/qa")
async def qa(arxiv_link: str = Form(...), question: str = Form(...)):
    if arxiv_link not in lru:
        chunks_db = await index_arxiv(arxiv_link)
    else:
        chunks_db = lru[arxiv_link]
    top_chunks = await topk_chunks(question, chunks_db, OPENAI_KEY)
    answer = await refine_qa(question, top_chunks, OPENAI_KEY)
    return {"answer": answer}


@app.get("/")
async def main():
    return {"Hello": "World"}
