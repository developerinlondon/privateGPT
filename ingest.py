import os
import glob
from typing import List
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS


load_dotenv()


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    if os.path.isdir(file_path):
        print("skipping directory %s\n" % file_path)
        return False
    if file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    else:
#    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf8")
    return loader.load()[0]


def load_documents(root_dirs: str) -> List[Document]:
    # Loads all documents from source documents directory
   # txt_files = glob.glob(os.path.join(source_dir, "**/*.txt"), recursive=True)
   # pdf_files = glob.glob(os.path.join(source_dir, "**/*.pdf"), recursive=True)
   # csv_files = glob.glob(os.path.join(source_dir, "**/*.csv"), recursive=True)
   # all_files = txt_files + pdf_files + csv_files
    #all_files = glob.glob(os.path.join(source_dir, "**/*"), recursive=True)
    #return [load_single_document(file_path) for file_path in all_files]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = []
    docs = []
    for root_dir in root_dirs:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                try: 
                    loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                    docs = loader.load_and_split()
                    docs = text_splitter.split_documents(docs)
                    for d in docs:
                        d.metadata['repo'] = root_dir
                        d.metadata['filename'] = file
                    texts.extend(docs)
                except Exception as e: 
                    print(file, e)
    return texts      
    

def main():
    # Load environment variables
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
    llama_embeddings_model = os.environ.get('LLAMA_EMBEDDINGS_MODEL')
    model_n_ctx = os.environ.get('MODEL_N_CTX')

    # Load documents and split in chunks
    print(f"Loading documents from {source_directory}")

    texts = load_documents(source_directory)
    
   # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
   # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
   # texts = text_splitter.split_documents(documents)
   # print(f"Loaded {len(documents)} documents from {source_directory}")
    print(f"Split into {len(texts)} chunks of text (max. 1000 tokens each)")
      
    # Create embeddings
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    
    # Create and store locally vectorstore
    db = Chroma.from_documents(texts, llama, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None


if __name__ == "__main__":
    main()
