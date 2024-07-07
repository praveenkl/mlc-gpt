import requests
from bs4 import BeautifulSoup
import re
import newspaper
import json
import time
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
from groq import Groq

seen_urls = set()
excluded_domains = ['youtube.com', 'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com']

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
openai.api_key = os.environ["OPENAI_API_KEY"]
Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

db = chromadb.PersistentClient(path="../data/news/chroma_db")
index_storage_dir = "../data/news/storage"

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

def summarize(article_text):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    system_prompt = '''
You are a helpful assistant highly skilled at summarizing a news article by extracting relevant information about the major league cricket tournament from that article. 

You will be given a news article. Your job is to extract relevant information about the major league cricket tournament from that article and summarizing it. 

Make sure to use only the information present in the news article to come up with your summary.
'''

    chat_completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
        },
        {
                "role": "user",
                "content": article_text,
        }
    ],
    temperature=1,
    max_tokens=1024,
    top_p=1,
    stream=False,
    stop=None,
    )

    response = chat_completion.choices[0].message.content
    # Remove any introductory text from the response
    response = response[response.find(':')+1:].strip()
    return response

def add_to_year_index(year, documents):
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
    refresh_info = index.refresh_ref_docs(documents)
    index.storage_context.persist(persist_dir=f'{index_storage_dir}/{year}')

def index_articles(articles):
    documents_2023 = []
    documents_2024 = []
    num_empty_articles = 0
    for article in articles:
        id = article['uri']
        b = article['body']
        date = article['date']
        if len(date) > 0 and len(b) > 0 and "major league cricket" in b[:500].lower():
            try:
                summary = summarize(b)
                # print(summary)
                time.sleep(5)
            except Exception as e:
                print(f'Error summarizing article {id}: {e}')
                continue
            d = schema.Document(doc_id=id, text=summary)
            year = int(date.split()[-1])
            if year <= 2023:
                documents_2023.append(d)
            elif year == 2024:
                documents_2024.append(d)
        else:
            num_empty_articles += 1
    print(f"Number of empty or skipped articles: {num_empty_articles}")

    add_to_year_index(2023, documents_2023)
    add_to_year_index(2024, documents_2024)
    print(f"Number of documents indexed: {len(documents_2023) + len(documents_2024)}")

    # Write the id and text (summary) of the documents to a file
    timestamp = time.strftime('%Y%m%d%H%M%S')
    year = time.strftime('%Y')
    month = time.strftime('%m')
    output_folder = f'raw/{year}/{month}'
    os.makedirs(output_folder, exist_ok=True)
    summary_output_filename = f'{output_folder}/summaries_{timestamp}.json'
    with open(summary_output_filename, 'w') as f:
        json.dump([{'uri': d.doc_id, 'text': d.text} for d in documents_2023 + documents_2024], f, ensure_ascii=False, indent=4)

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
    except Exception as e:
        print(f'Error: {e}')
    
    try:
        articles = crawl_and_save_articles(new_article_urls)
        index_articles(articles)
    except Exception as e:
        print(f'Error: {e}')
   
    # Read all the files starting with "articles" in the raw folder and index them
    # articles = []
    # for root, dirs, files in os.walk('raw'):
    #     for file in files:
    #         if file.startswith('articles'):
    #             with open(os.path.join(root, file)) as f:
    #                 articles.extend(json.load(f))
