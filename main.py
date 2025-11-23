# import os
# import io
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from dotenv import load_dotenv

# # PDF processing
# import pymupdf

# # --- NEW CORRECTED LANGCHAIN IMPORTS ---
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# # --- END OF CORRECTED IMPORTS ---

# # --- 1. Load Environment Variables & API Key ---
# load_dotenv()

# # --- 2. Initialize FastAPI App ---
# app = FastAPI(
#     title="PDF Q&A RAG Service",
#     description="An API to upload a PDF, process it, and ask questions using RAG.",
#     version="1.0.0"
# )

# # --- 3. Add CORS Middleware ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins (for local development)
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all HTTP methods
#     allow_headers=["*"],  # Allows all headers
# )

# # --- 4. Global In-Memory Store ---
# pdf_store = {
#     "retriever": None
# }

# # --- 5. Initialize AI Models ---
# try:
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
# except Exception as e:
#     print(f"Error initializing Google AI models: {e}")
#     # Handle error appropriately
    
# # --- 6. Pydantic Models for Request Body ---
# class AskPayload(BaseModel):
#     question: str

# # --- 7. API Endpoints ---

# @app.get("/", tags=["General"])
# async def read_root():
#     return {"message": "PDF Q&A RAG service is running."}

# @app.post("/upload", tags=["PDF Processing"])
# async def upload_pdf(file: UploadFile = File(...)):
#     """
#     Uploads a PDF, extracts text, creates embeddings, and stores them 
#     in an in-memory FAISS vector store.
#     """
#     global pdf_store
    
#     if file.content_type != "application/pdf":
#         raise HTTPException(status_code=400, detail="File must be a PDF.")
    
#     try:
#         # Read file into memory
#         contents = await file.read()
        
#         # Use pymupdf to open PDF from bytes
#         pdf_document = pymupdf.open("pdf", contents)
        
#         full_text = ""
#         for page_num in range(len(pdf_document)):
#             page = pdf_document.load_page(page_num)
#             full_text += page.get_text()
            
#         pdf_document.close()

#         if not full_text:
#             raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

#         # --- RAG Pipeline: Chunking ---
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=100,
#             length_function=len
#         )
#         chunks = text_splitter.split_text(full_text)

#         if not chunks:
#              raise HTTPException(status_code=400, detail="Could not create text chunks.")

#         # --- RAG Pipeline: Embedding & Storing ---
#         # Create FAISS vector store from the text chunks
#         vector_store = FAISS.from_texts(chunks, embeddings)
        
#         # --- RAG Pipeline: Create Retriever ---
#         pdf_store["retriever"] = vector_store.as_retriever(search_kwargs={"k": 5})

#         return {
#             "filename": file.filename,
#             "status": "processed",
#             "total_chunks": len(chunks)
#         }

#     except Exception as e:
#         print(f"Error during PDF processing: {e}")
#         raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# @app.post("/ask", tags=["Q&A"])
# async def ask_question(payload: AskPayload):
#     """
#     Asks a question to the processed PDF using the stored RAG retriever.
#     """
#     global pdf_store, llm
    
#     retriever = pdf_store.get("retriever")
    
#     if retriever is None:
#         raise HTTPException(status_code=400, detail="No PDF has been processed. Please upload a PDF first.")
        
#     if not payload.question:
#         raise HTTPException(status_code=400, detail="No question provided.")

#     try:
#         # --- RAG Prompt Template ---
#         template = """
#         You are a helpful assistant. Answer the user's question based *only* on the
#         following context provided from a PDF document.
        
#         If the information to answer the question is not in the context,
#         say: "I cannot find the answer in the provided document."
        
#         Do not make up information.
        
#         Context:
#         {context}
        
#         Question:
#         {question}
        
#         Helpful Answer:
#         """
#         prompt = ChatPromptTemplate.from_template(template)

#         # --- RAG Chain ---
#         rag_chain = (
#             {"context": retriever, "question": RunnablePassthrough()}
#             | prompt
#             | llm
#             | StrOutputParser()
#         )
        
#         # Invoke the chain with the user's question
#         answer = rag_chain.invoke(payload.question)
        
#         return {"answer": answer}

#     except Exception as e:
#         print(f"Error during Q&A: {e}")
#         raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# import os
# import io
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# # PDF processing
# import pymupdf

# # --- NEW OLLAMA LANGCHAIN IMPORTS ---
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings
# # --- END OF OLLAMA IMPORTS ---

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# # --- 1. Initialize FastAPI App ---
# app = FastAPI(
#     title="PDF Q&A RAG Service (Ollama Local)",
#     description="An API to upload a PDF and ask questions using a local LLM.",
#     version="1.0.0"
# )

# # --- 2. Add CORS Middleware ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all HTTP methods
#     allow_headers=["*"],  # Allows all headers
# )

# # --- 3. Global In-Memory Store ---
# pdf_store = {
#     "retriever": None
# }

# # --- 4. Initialize AI Models (Now using Ollama) ---
# try:
#     # This assumes Ollama is running. We are using the 'llama3:8b' model.
#     embeddings = OllamaEmbeddings(model="llama3:8b")
#     llm = Ollama(model="llama3:8b")
# except Exception as e:
#     print(f"Error initializing Ollama models. Is Ollama running? {e}")
#     # Handle error appropriately
    
# # --- 5. Pydantic Models for Request Body ---
# class AskPayload(BaseModel):
#     question: str

# # --- 6. API Endpoints ---

# @app.get("/", tags=["General"])
# async def read_root():
#     return {"message": "PDF Q&A RAG service (Ollama Local) is running."}

# @app.post("/upload", tags=["PDF Processing"])
# async def upload_pdf(file: UploadFile = File(...)):
#     """
#     Uploads a PDF, extracts text, creates embeddings, and stores them 
#     in an in-memory FAISS vector store.
#     """
#     global pdf_store
    
#     if file.content_type != "application/pdf":
#         raise HTTPException(status_code=400, detail="File must be a PDF.")
    
#     try:
#         # Read file into memory
#         contents = await file.read()
        
#         # Use pymupdf to open PDF from bytes
#         pdf_document = pymupdf.open("pdf", contents)
        
#         full_text = ""
#         for page_num in range(len(pdf_document)):
#             page = pdf_document.load_page(page_num)
#             full_text += page.get_text()
            
#         pdf_document.close()

#         if not full_text:
#             raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

#         # --- RAG Pipeline: Chunking ---
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=100,
#             length_function=len
#         )
#         chunks = text_splitter.split_text(full_text)

#         if not chunks:
#              raise HTTPException(status_code=400, detail="Could not create text chunks.")

#         # --- RAG Pipeline: Embedding & Storing ---
#         # Create FAISS vector store from the text chunks
#         vector_store = FAISS.from_texts(chunks, embeddings)
        
#         # --- RAG Pipeline: Create Retriever ---
#         pdf_store["retriever"] = vector_store.as_retriever(search_kwargs={"k": 5})

#         return {
#             "filename": file.filename,
#             "status": "processed",
#             "total_chunks": len(chunks)
#         }

#     except Exception as e:
#         print(f"Error during PDF processing: {e}")
#         raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# @app.post("/ask", tags=["Q&A"])
# async def ask_question(payload: AskPayload):
#     """
#     Asks a question to the processed PDF using the stored RAG retriever.
#     """
#     global pdf_store, llm
    
#     retriever = pdf_store.get("retriever")
    
#     if retriever is None:
#         raise HTTPException(status_code=400, detail="No PDF has been processed. Please upload a PDF first.")
        
#     if not payload.question:
#         raise HTTPException(status_code=400, detail="No question provided.")

#     try:
#         # --- RAG Prompt Template ---
#         template = """
#         You are a helpful assistant. Answer the user's question based *only* on the
#         following context provided from a PDF document.
        
#         If the information to answer the question is not in the context,
#         say: "I cannot find the answer in the provided document."
        
#         Do not make up information.
        
#         Context:
#         {context}
        
#         Question:
#         {question}
        
#         Helpful Answer:
#         """
#         prompt = ChatPromptTemplate.from_template(template)

#         # --- RAG Chain ---
#         rag_chain = (
#             {"context": retriever, "question": RunnablePassthrough()}
#             | prompt
#             | llm
#             | StrOutputParser()
#         )
        
#         # Invoke the chain with the user's question
#         answer = rag_chain.invoke(payload.question)
        
#         return {"answer": answer}

#     except Exception as e:
#         print(f"Error during Q&A: {e}")
#         raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


#                   vertexai

import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# PDF processing
import fitz as pymupdf

# --- NEW VERTEX AI LANGCHAIN IMPORTS ---
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
# --- END OF VERTEX AI IMPORTS ---

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Load Environment Variables ---
load_dotenv()

# --- 2. Initialize FastAPI App ---
app = FastAPI(
    title="PDF Q&A RAG Service (Vertex AI)",
    description="An API to upload a PDF and ask questions using Google Vertex AI.",
    version="2.0.2" # Version bump
)

# --- 3. Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# --- 4. Global In-Memory Store & Model Placeholders ---
pdf_store = {
    "retriever": None
}
llm = None
embeddings = None
PROJECT_ID = "gen-lang-client-0172745415" # Your Project ID

# --- 5. Initialize AI Models (Now using Vertex AI) ---
try:
    print(f"Initializing Vertex AI with Project ID: {PROJECT_ID}")
    
    llm = VertexAI(
        model_name="gemini-2.5-pro", 
        temperature=0.3, 
        project=PROJECT_ID
    )
    
    # --- THE ONLY CHANGE IS THIS LINE ---
    embeddings = VertexAIEmbeddings(
        model_name="text-multilingual-embedding-002", # Changed from gecko@003
        project=PROJECT_ID
    )
    # --- END OF CHANGE ---
    
    print("Vertex AI models initialized successfully.")

except Exception as e:
    print(f"--- CRITICAL ERROR: FAILED TO INITIALIZE VERTEX AI MODELS ---")
    print(f"Error: {e}")
    print("Please check your Google Cloud Project ID, billing, and API permissions.")
    
# --- 6. Pydantic Models for Request Body ---
class AskPayload(BaseModel):
    question: str

# --- 7. API Endpoints ---

@app.get("/", tags=["General"])
async def read_root():
    return {"message": "PDF Q&A RAG service (Vertex AI) is running."}

@app.post("/upload", tags=["PDF Processing"])
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_store, embeddings
    
    if embeddings is None:
        raise HTTPException(
            status_code=500, 
            detail="AI embedding model failed to load. Check server logs for errors."
        )
    
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF.")
    
    try:
        contents = await file.read()
        pdf_document = pymupdf.open("pdf", contents)
        
        full_text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            full_text += page.get_text()
        pdf_document.close()

        if not full_text:
            raise HTTPException(status_code=4.00, detail="Could not extract text from PDF.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(full_text)

        if not chunks:
             raise HTTPException(status_code=400, detail="Could not create text chunks.")

        vector_store = FAISS.from_texts(chunks, embeddings)
        
        pdf_store["retriever"] = vector_store.as_retriever(search_kwargs={"k": 5})

        return {
            "filename": file.filename,
            "status": "processed",
            "total_chunks": len(chunks)
        }

    except Exception as e:
        print(f"Error during PDF processing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/ask", tags=["Q&A"])
async def ask_question(payload: AskPayload):
    global pdf_store, llm
    
    retriever = pdf_store.get("retriever")
    
    if llm is None:
        raise HTTPException(
            status_code=500, 
            detail="AI chat model failed to load. Check server logs for errors."
        )
        
    if retriever is None:
        raise HTTPException(status_code=400, detail="No PDF has been processed. Please upload a PDF first.")
        
    if not payload.question:
        raise HTTPException(status_code=400, detail="No question provided.")

    try:
        template = """
        You are a helpful assistant. Answer the user's question based *only* on the
        following context provided from a PDF document.
        
        If the information to answer the question is not in the context,
        say: "I cannot find the answer in the provided document."
        
        Do not make up information.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Helpful Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = rag_chain.invoke(payload.question)
        
        return {"answer": answer}

    except Exception as e:
        print(f"Error during Q&A: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

