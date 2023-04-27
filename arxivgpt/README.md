# arxiv Q&A
Given a link to a arxiv paper use chatgpt to ask the paper questions.

![](https://github.com/yonigottesman/llm_playground/blob/main/arxivgpt/images/screenshot.png)

## Playground
instead of running the client and server you can just run `server/offline_run.py` which basically does this:
1. Downlaod pdf from arxiv
2. `index_long_document` splits the pdf to chunks, creates embeddings for each chunk, and stores them in db which is basically a np.array.
3. get topk chunks for a question will create an embedding for the question and find the topk chunks that are most similar to the question.
4. refine_qa will use the topk chunks to query chatgpt iteratively and refine the answer.

~~~python
pdf = asyncio.run(get_pdf("https://arxiv.org/abs/2010.11929"))
large_document = " ".join(p.extract_text() for p in pdf.pages)
db = asyncio.run(index_long_document(large_document, OPENAI_KEY, chunk_size=500))

question = "On what dataset was the model trained?"
result = asyncio.run(topk_chunks(question, db, OPENAI_KEY))
answer = asyncio.run(refine_qa(question, result, OPENAI_KEY))
print(answer)
~~~

## Running Fronend and Backend
* Run backend from `server` folder with: (after poetry install)
    ~~~bash
    OPENAI_KEY=<openai_key> uvicorn main:app
    ~~~
* Run frontend from `client` folder with:
    ~~~
    npm run build
    REACT_APP_BACKEND_IP="http://127.0.0.1:8000" npm start
    ~~~
