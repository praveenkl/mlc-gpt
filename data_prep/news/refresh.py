import requests
from bs4 import BeautifulSoup
import re
import newspaper
import json
import time
import os
import sys
import logging

from llama_index import ServiceContext, StorageContext, VectorStoreIndex, schema
from llama_index.embeddings import OpenAIEmbedding
from llama_index.vector_stores import ChromaVectorStore
from llama_index.llms import OpenAI

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import openai

seen_urls = set()
excluded_domains = ['youtube.com', 'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com']

def google_news(seed_url):
    response = requests.get(seed_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    urls = []
    skipped_urls = []
    for link in soup.find_all('a', attrs={'href': re.compile(r"\/url\?q=(.*?)&sa=U")}):
        candidate = re.findall(r"\/url\?q=(.*?)&sa=U", link.get('href'))[0]
        if re.search("google.com", candidate) or not candidate.startswith('http'):
            continue
        if not any([re.search(domain, candidate) for domain in excluded_domains]) and candidate not in seen_urls:
                seen_urls.add(candidate)
                urls.append(candidate)
        else:
            skipped_urls.append(candidate)
            # print(f'Skipping url: {candidate}')
    return urls, skipped_urls

def collect_articles(full_refresh=False):
    seed_url = 'https://www.google.com/search?q=%22major+league+cricket%22&tbm=nws&source=lmns&hl=en&start={start_idx}'
    if not full_refresh:
        seed_url += '&tbs=qdr:w'
    full_article_list = []
    for i in range(0, 1000, 10):
        article_list, skipped_list = google_news(seed_url.format(start_idx=i))
        if len(article_list) == 0 and len(skipped_list) == 0:
            print(f'No more articles found on google news results page {int(i/10 + 1)}')
            break
        full_article_list.extend(article_list)
        print(f'Collected {len(article_list)} urls from google news results page {int(i/10 + 1)}')
        time.sleep(1)
    print(f'Total urls collected: {len(full_article_list)}')
    return full_article_list

def crawl_and_save_articles(new_article_urls):
    articles = []
    for url in new_article_urls:
        try:
            article = newspaper.article(url)
            if article.publish_date:
                date = article.publish_date.ctime()
            else:
                date = ""
        except:
            print(f'Error crawling article: {url}')
            continue
        b = article.text
        if len(b) > 0 and "major league cricket" in b[:500].lower():
            articles.append({'uri': url.rstrip(), 'title': article.title, 'date': date, 'body': article.text})
        else:
            # print(f'Skipping article: {url}')
            continue
        if len(articles) % 10 == 0:
            print(f'Crawled {len(articles)} articles')
        time.sleep(1)
    print(f'Total articles crawled: {len(articles)}')

    if len(articles) > 0:
        timestamp = time.strftime('%Y%m%d%H%M%S')
        year = time.strftime('%Y')
        month = time.strftime('%m')
        output_folder = f'raw/{year}/{month}'
        os.makedirs(output_folder, exist_ok=True)
        article_output_filename = f'{output_folder}/articles_{timestamp}.json'
        with open(article_output_filename, 'w') as f:
            json.dump(articles, f, ensure_ascii=False, indent=4)

    if len(new_article_urls) > 0:
        with open('raw/article_list.txt', 'a') as f:
            f.write('\n'.join(new_article_urls) + '\n')
        print('Article list updated')
    return articles

def index_articles(articles):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    openai.api_key = os.environ["OPENAI_API_KEY"]
    embed_model = OpenAIEmbedding()
    llm = OpenAI(model="gpt-4")
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

    db = chromadb.PersistentClient(path="../../chroma_db")
    chroma_collection = db.get_or_create_collection("mlc_articles")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    if os.path.exists("./storage"):    
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="./storage")
    else:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=[],
        service_context=service_context,
        storage_context=storage_context,
    )

    documents = []
    num_empty_articles = 0
    for article in articles:
        id = article['uri']
        b = article['body']
        if len(b) > 0 and "major league cricket" in b[:500].lower():
            d = schema.Document(doc_id=id, text=b)
            documents.append(d)
        else:
            num_empty_articles += 1
    print(f"Number of empty or skipped articles: {num_empty_articles}")

    refresh_info = index.refresh_ref_docs(documents)
    index.storage_context.persist()
    print(f"Number of documents indexed: {len(documents)}")

if __name__ == '__main__':
    new_article_urls = []
    existing_article_urls = []
    try:
        with open('raw/article_list.txt') as f:
            print('Aricle list found. Checking for new articles.')
            existing_article_urls = [line.rstrip() for line in f.readlines()]
            seen_urls = set(existing_article_urls)
        new_article_urls = collect_articles(full_refresh=False)
    except FileNotFoundError:
        print('Article list not found. Collecting full set of articles.')
        new_article_urls = collect_articles(full_refresh=True)
    
    articles = crawl_and_save_articles(new_article_urls)
    index_articles(articles)