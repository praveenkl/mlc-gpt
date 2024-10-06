import os
import sys
import logging

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI

import chromadb
import openai

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
openai.api_key = os.environ["OPENAI_API_KEY"]
Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

db = chromadb.PersistentClient(path="./data/news/chroma_db")
index_storage_dir = "./data/news/storage"

def delete_from_year_index(year, article_ids):
    chroma_collection = db.get_or_create_collection(f'mlc_articles_{year}')
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    if os.path.exists(f'{index_storage_dir}/{year}'):
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=f'{index_storage_dir}/{year}')
    else:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=[],
        storage_context=storage_context,
    )
    for article_id in article_ids: 
        index.delete_ref_doc(article_id)
    index.storage_context.persist(persist_dir=f'{index_storage_dir}/{year}')

if __name__ == '__main__':
    # Create a list of uri's to delete
    uri_list = [
        "https://isportindia.com/news/pat-cummins-will-play-for-this-team-in-major-league-cricket-mlc-season-two",
        "https://www.isportindia.com/news/kkr-team-faces-another-defeat-two-south-african-players-worked-together-to-make-it-all",
        "https://www.isportindia.com/news/pat-cummins-showed-the-way-out-to-super-kings-did-not-allow-18-runs-to-be-scored-in-the-last-over-delhi-capitals-batsman-hit-a-century",
        "https://www.isportindia.com/news/ipl-2022-rajasthan-royals-will-play-against-mumbai-indians-today",
        "https://isportindia.com/news/south-african-batsman-joins-csk-franchise-signs-deal-ahead-of-major-t20-league",
        "https://en.wikipedia.org/wiki/2024_Major_League_Cricket_season",
        "https://en.wikipedia.org/wiki/Texas_Super_Kings",
        "https://en.wikipedia.org/wiki/Cricket_in_the_United_States",
        "https://en.wikipedia.org/wiki/Washington_Freedom_(cricket)",
        "https://en.wikipedia.org/wiki/Wikipedia:WikiProject_Major_League_Cricket",
        "https://www.expressvpn.com/stream-sports/cricket/major-league-cricket/"
    ]
    # Delete the articles from the index
    delete_from_year_index(2023, uri_list)
    delete_from_year_index(2024, uri_list)


    

