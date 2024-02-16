from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_community.llms import CTransformers,HuggingFaceHub
from langchain.chains import RetrievalQA
import chainlit as cl
import asyncio
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import re
import json
import config as cfg

DB_FAISS_PATH = "vectorstore/db_constiution"
print(DB_FAISS_PATH)
custom_prompt_template  = """ Use the following pieces of information to answer the users question.
                    If you don't know the answer, please just say that you dont know the answer dont try
                    to make up an answer.

                    context: {context}
                    Question: {question}

                    only returns the helpful on json format answer as key answer below and nothing else.
                    #Helpful answer:

                
                    """

# response_schema = ResponseSchema(anme="")

def set_custome_prompt():
    """
    Prompt template for qa retrival.
    """
    prompt = PromptTemplate(template= custom_prompt_template, input_variables=['context','question'])
    return prompt

def load_llm():

    llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", 
                         huggingfacehub_api_token=cfg.HUGGING_FACE_API_TOKEN,
                         model_kwargs={
                             "temperature":0.5,
                             "max_length":64,
                             "max_new_tokens":512
                         })

    return llm

def retrival_qa_chain(llm, prompt,db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever= db.as_retriever(search_kawrgs={"k":2}),
        return_source_documents = True,
        chain_type_kwargs= {'prompt':prompt}

    )

    return qa_chain


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custome_prompt()
    qa = retrival_qa_chain(llm, qa_prompt, db)
    return qa


def final_result(query):
    qa_result = qa_bot()
    response = qa_result({"query":query})

    return response

def format_output (llm_output):
    match = re.search(r'{\n\s*"answer":\s*".*"\n\s*}', llm_output, re.DOTALL)
    if match:
        json_str = match.group(0)
        # Parse the JSON string
        answer_data = json.loads(json_str)
        # Extract the "answer" field
        answer = answer_data["answer"]
        return answer
    else:
        return "Answer not found"

async def run_chat():
    chain = qa_bot()
    prompt = input("Input query:  ")

    res = await chain.ainvoke(prompt)
    sources = res["source_documents"][0].dict()
    answer =  format_output(res["result"])
    answer = answer  + f'\n {sources["metadata"]["source"]} page no: {sources["metadata"]["page"]}'
    print(answer)


# if __name__ == "__main__":
#     asyncio.run(run_chat())

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg= cl.Message(content= "Starting the bot ..." )
    await msg.send()
    msg.content = "hi, Welcome to Nepal Constitution qa bot"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"][0].dict()
    answer =  format_output(res["result"])
 
    answer = answer  + f'\n {sources["metadata"]["source"]} page no: {sources["metadata"]["page"]}'


    await cl.Message(content=answer).send()