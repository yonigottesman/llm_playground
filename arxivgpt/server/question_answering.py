import openai

CHATGPT_MODEL = "gpt-3.5-turbo"


QA_PROMT = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and no prior knowledge, answer the question:\n {question}\n"
    "If the answer cannot be found in the context, write 'I could not find an answer.'"
)


REFINE_PROMPT = (
    "Given the following input structure:\n"
    "\n----\n"
    "QUESTION: <existing question>\n"
    "EXISTING_ANSWER: <existing answer>.\n"
    "NEW_CONTEXT: <new context>.\n"
    "\n----\n"
    "Given the new context, refine the existing answer to better answer the question.\n"
    "Only answer the given question without additional information, or answers to other questions.\n"
    "If the existing answer has information that answers the question dont remove it.\n"
    "If the context is useful. Return the new answer to the question.\n\n"
)


REFINE_FEW_SHOT_EXAMPLES = (
    {
        "role": "user",
        "content": "QUESTION: what is the color of the dog?\n"
        "EXISTING_ANSWER: the dog is brown.\n"
        "NEW_CONTEXT: the cat is black.\n",
    },
    {"role": "assistant", "content": "ANSWER: the dog is brown."},
    {
        "role": "user",
        "content": "QUESTION: what is the color of the dog?\n"
        "EXISTING_ANSWER: the dog is brown.\n"
        "NEW_CONTEXT: the dog has white spots.\n",
    },
    {"role": "assistant", "content": "ANSWER: the dog is brown with white spots."},
)

REFINE_FORMAT = "QUESTION: {question}\nEXISTING_ANSWER: {existing_answer}\nNEW_CONTEXT: {context}\n"


async def refine_qa(question: str, top_documents: list[str], open_ai_key: str) -> str:
    # Initial question and answer
    messages = [
        {"role": "system", "content": "You answer questions about a paper from arxiv"},
        {"role": "user", "content": QA_PROMT.format(context_str=top_documents[0], question=question)},
    ]
    response = await openai.ChatCompletion.acreate(
        messages=messages,
        model=CHATGPT_MODEL,
        max_tokens=2000,
        temperature=0,
        api_key=open_ai_key,
    )
    current_answer = response["choices"][0]["message"]["content"]
    # print(current_answer)

    # Refine the answer on the rest of the documents
    for res in top_documents[1:]:
        messages = [
            {"role": "system", "content": "You answer questions about a paper from arxiv"},
            {"role": "user", "content": REFINE_PROMPT},
            *REFINE_FEW_SHOT_EXAMPLES,
            {
                "role": "user",
                "content": REFINE_FORMAT.format(context=res, question=question, existing_answer=current_answer),
            },
        ]

        response = await openai.ChatCompletion.acreate(
            messages=messages,
            model=CHATGPT_MODEL,
            max_tokens=2000,
            temperature=0,
            api_key=open_ai_key,
        )
        current_answer = response["choices"][0]["message"]["content"]
        # print(current_answer)
    return current_answer
