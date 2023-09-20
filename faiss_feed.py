import argparse
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

def main(data_path):
    text_loader_kwargs = {'autodetect_encoding': True}
    loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(docs)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cuda"}

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    # storing embeddings in the vector store
    vectorstore_0 = FAISS.from_documents(all_splits, embeddings)

    # Save the FAISS index
    vectorstore_0.storage_context.persist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load raw data and save FAISS index.")
    parser.add_argument("data_path", type=str, help="Path to access the raw data.")
    
    args = parser.parse_args()
    main(args.data_path)
