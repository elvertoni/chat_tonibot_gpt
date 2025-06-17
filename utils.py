import os
import tempfile
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    os.remove(temp_file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
    )
    chunks = splitter.split_documents(docs)
    return chunks


def load_vector_store(persist_directory):
    # Ensure the directory exists before trying to load from it
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(),
        )
    return None


def add_to_vector_store(chunks, persist_directory, vector_store=None):
    # Ensure the directory exists before trying to write to it
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)

    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory,
        )
    return vector_store


def reset_vector_store(persist_directory):
    shutil.rmtree(persist_directory, ignore_errors=True)
