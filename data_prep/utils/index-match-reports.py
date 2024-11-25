import os
import sys
import logging

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from llama_index.core import Settings, StorageContext, VectorStoreIndex, schema
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI

import chromadb
import openai

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
openai.api_key = os.environ["OPENAI_API_KEY"]
Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

db = chromadb.PersistentClient(path="./news/chroma_db")
index_storage_dir = "./news/storage"

def add_to_year_index(year, documents):
    chroma_collection = db.get_or_create_collection(f'milc_articles_{year}')
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    if os.path.exists(f'{index_storage_dir}/{year}'):
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=f'{index_storage_dir}/{year}')
    else:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=[],
        storage_context=storage_context,
    )
    refresh_info = index.refresh_ref_docs(documents)
    index.storage_context.persist(persist_dir=f'{index_storage_dir}/{year}')

def index_match_reports(reports):
    documents_2024 = []
    for report in reports:
        id = report['id']
        b = report['body']
        d = schema.Document(doc_id=id, text=b)
        d.metadata = {"title": "", "date": "", "uri": id}
        d.excluded_embed_metadata_keys = ["uri", "date", "title"]
        d.excluded_llm_metadata_keys = ["uri", "date", "title"]
        documents_2024.append(d)
    add_to_year_index(2024, documents_2024)
    print(f"Number of documents indexed: {len(documents_2024)}")
    return reports

if __name__ == '__main__':
    reports = []
    for file in os.listdir("match_reports"):
        with open(f"match_reports/{file}", "r") as f:
            reports.append({"id": file.split(".")[0], "body": f.read()})
    indexed_reports = index_match_reports(reports)
