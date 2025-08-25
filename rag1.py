from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory



# API KEY - BHEJI jayegi. So.
# PDF ki text string - Bheji jayegi.
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite', google_api_key=os.environ["GOOGLE_API_KEY"])


loader = WebBaseLoader(
    web_path=("https://en.wikisource.org/wiki/Analysis_and_Assessment_of_Gateway_Process")
)

docs = loader.load()

def load_text(text: str, api_key: str):
    count = 0
    divided = "<sep>"
    for i in range(len(text)):
        if count == 2000:
            divided += "<sep>"
            count = 0
        else:
            count += 1
            divided += text[i]
    prompt = f"""You are a preprocessing assistant. Your task is to rewrite text chunks into self-contained units that can be understood without additional context. Do not add new information; only restate or clarify using the text itself."

    The following text contains multiple chunks separated by the token <sep>.  

    Task:  
    - For each chunk, rewrite it so it is fully self-contained and understandable without other chunks.  
    - Do not add new facts, only restate missing references.  
    - Keep meaning and details intact.  
    - Output each rewritten chunk in order, labeled clearly.  
    - KEEP THE NUMBER OF CHUNKS SAME AND INTACT.

    Text: {divided}
    Number of Chunks: {len(divided.split("<sep>"))}
    Example output(JSON):
    {{
        "0":"Chunk1",
        "1":"Chunk2",
    }}
    """
    model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite', google_api_key=os.environ["GOOGLE_API_KEY"])
    contextualized_data = model.invoke(prompt)




    json_string = "{" + contextualized_data.content[12:]
    json_string2 = json_string[:-7] + '"}'


    data = json.loads(json_string2)

    original_data = divided.split("<sep>")
    meta = list(data.values())

    result = []
    max_len = max(len(original_data), len(meta))
    for i in range(max_len):
        x = original_data[i] if i < len(original_data) else ""
        y = meta[i] if i < len(meta) else ""
        result.append(f"{x}. Document Meaning : {y}")


    docs = [Document(page_content=item) for item in result]
    return docs



def embed(api_key:str, docs:list):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_documents(docs, embeddings)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})
    return retriever



def chain(api_key:str, retriever):
    prompt = PromptTemplate(template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.
    Context: {context}
    Question: {question}
                            """, input_variables=['context', 'question'])


    ## Prompt Template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever=create_history_aware_retriever(model,retriever,contextualize_q_prompt)

    question_answer_chain=create_stuff_documents_chain(model,prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
    return rag_chain

# This will probably be done in the app.py
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

conversational_rag_chain.invoke(
    {"input": "is structure of universe covered? Explain it in detail"},
    config={
        "configurable": {"session_id": "1"}
    },  # constructs a key "abc123" in `store`.
)["answer"]