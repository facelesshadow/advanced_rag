# rag.py
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
import json, os

def load_text_to_docs(text: str, api_key: str) -> list[Document]:
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
    model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite', google_api_key=api_key)
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

def create_retriever(docs: list[Document], api_key: str):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})

def build_rag_chain(api_key: str, retriever):
    model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite', google_api_key=api_key)

    ## Prompt Template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. If the query is not about the document, answer accordingly."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),   # 👈 add this
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
