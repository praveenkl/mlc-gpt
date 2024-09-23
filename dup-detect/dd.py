import os
import json
import pickle
import datetime
from datasketch import MinHash, MinHashLSH
from nltk.tokenize import word_tokenize
import nltk

# Ensure the required NLTK data files are downloaded
nltk.download('punkt', quiet=True)

def read_articles_and_summaries(root_dir):
    articles = []
    article_ids = []
    article_id = 0

    # Build a global mapping from uri to summary text
    summaries_dict = {}

    # First, read all summaries and build the mapping
    for year in os.listdir(root_dir):
        year_path = os.path.join(root_dir, year)
        if os.path.isdir(year_path):
            for month in os.listdir(year_path):
                month_path = os.path.join(year_path, month)
                if os.path.isdir(month_path):
                    for filename in os.listdir(month_path):
                        if filename.startswith('summaries_') and filename.endswith('.json'):
                            summary_file_path = os.path.join(month_path, filename)
                            with open(summary_file_path, 'r', encoding='utf-8') as f:
                                try:
                                    summaries_list = json.load(f)
                                    # Build a mapping from uri to summary text
                                    for summary in summaries_list:
                                        uri = summary.get('uri')
                                        text = summary.get('text', '')
                                        if uri:
                                            summaries_dict[uri] = text
                                except json.JSONDecodeError as e:
                                    print(f"Error decoding JSON from {summary_file_path}: {e}")

    # Now, read all articles and match them with summaries using uri
    for year in os.listdir(root_dir):
        year_path = os.path.join(root_dir, year)
        if os.path.isdir(year_path):
            for month in os.listdir(year_path):
                month_path = os.path.join(year_path, month)
                if os.path.isdir(month_path):
                    for filename in os.listdir(month_path):
                        if filename.startswith('articles_') and filename.endswith('.json'):
                            article_file_path = os.path.join(month_path, filename)
                            # Read articles
                            with open(article_file_path, 'r', encoding='utf-8') as f:
                                try:
                                    data = json.load(f)
                                except json.JSONDecodeError as e:
                                    print(f"Error decoding JSON from {article_file_path}: {e}")
                                    continue

                            # Match articles with summaries using uri
                            for article in data:
                                uri = article.get('uri')
                                if uri and uri in summaries_dict:
                                    article['summary'] = summaries_dict[uri]
                                else:
                                    article['summary'] = ''
                                articles.append(article)
                                article_ids.append(article_id)
                                article_id += 1
    return articles, article_ids

def get_shingles(text, k=5):
    tokens = word_tokenize(text.lower())
    shingles = set()
    for i in range(len(tokens) - k + 1):
        shingle = ' '.join(tokens[i:i+k])
        shingles.add(shingle)
    return shingles

def create_minhash(shingles, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for shingle in shingles:
        m.update(shingle.encode('utf8'))
    return m

def find(u, parent):
    if parent[u] != u:
        parent[u] = find(parent[u], parent)
    return parent[u]

def union(u, v, parent):
    pu, pv = find(u, parent), find(v, parent)
    if pu != pv:
        parent[pu] = pv

def parse_date(date_str):
    if not date_str:
        return None
    # Expected date format: "Fri Jun 28 13:19:42 2024"
    try:
        return datetime.datetime.strptime(date_str.strip(), "%a %b %d %H:%M:%S %Y")
    except ValueError:
        return None

def main():
    root_dir = '/workspaces/mlc-gpt/data_prep/raw'  # Adjust this if your directory is named differently
    articles, article_ids = read_articles_and_summaries(root_dir)

    # Create MinHashLSH index for detecting duplicates
    lsh_duplicates = MinHashLSH(threshold=0.8, num_perm=128)

    # Keep track of MinHashes
    minhash_dict = {}

    # Initialize Disjoint Set Union (Union-Find) data structure
    parent = {idx: idx for idx in article_ids}

    # Build MinHash signatures and insert into LSH for duplicate detection
    for idx, article in zip(article_ids, articles):
        text = article.get('body', '').lower()
        shingles = get_shingles(text)
        m = create_minhash(shingles)
        lsh_duplicates.insert(idx, m)
        minhash_dict[idx] = m

    # Query for duplicates and union them
    for idx in article_ids:
        result = lsh_duplicates.query(minhash_dict[idx])
        for other_idx in result:
            if idx != other_idx:
                union(idx, other_idx, parent)

    # Collect articles into groups based on parent mapping
    groups = {}
    for idx in article_ids:
        root = find(idx, parent)
        groups.setdefault(root, []).append(idx)

    # Process groups to create unique articles
    unique_articles = []
    for group in groups.values():
        group_indices = group  # list of article_ids
        # Collect articles in the group
        group_articles = [articles[article_ids.index(idx)] for idx in group_indices]

        # Select the article with the oldest date
        oldest_article = None
        oldest_date = None
        for article in group_articles:
            date_str = article.get('date', '')
            parsed_date = parse_date(date_str)
            if parsed_date:
                if oldest_date is None or parsed_date < oldest_date:
                    oldest_date = parsed_date
                    oldest_article = article
            else:
                # If date is missing or invalid, consider it less preferred
                if oldest_date is None:
                    oldest_article = article  # Retain if no other articles with valid dates
        if oldest_article is None:
            oldest_article = group_articles[0]  # If all dates are invalid, pick the first one

        # Collect URIs of duplicate articles (excluding the retained one)
        retained_uri = oldest_article.get('uri')
        duplicate_uris = []
        for article in group_articles:
            uri = article.get('uri')
            if uri and uri != retained_uri:
                duplicate_uris.append(uri)
        if duplicate_uris:
            oldest_article['duplicate_uris'] = duplicate_uris
        else:
            oldest_article['duplicate_uris'] = []

        # Ensure 'summary' field is present
        if 'summary' not in oldest_article:
            oldest_article['summary'] = ''

        unique_articles.append(oldest_article)

    # Create output directory if it doesn't exist
    output_dir = 'unique_articles'
    os.makedirs(output_dir, exist_ok=True)

    # Group unique articles into lists of 1000 and write each list to a separate file
    batch_size = 1000
    for i in range(0, len(unique_articles), batch_size):
        batch = unique_articles[i:i + batch_size]
        batch_number = i // batch_size
        output_filename = os.path.join(output_dir, f'unique_articles_batch_{batch_number}.json')
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)
        print(f"Batch {batch_number} written to {output_filename}")

    # Create MinHashLSH index for unique articles
    lsh_unique = MinHashLSH(threshold=0.8, num_perm=128)
    unique_minhash_dict = {}

    for idx, article in enumerate(unique_articles):
        text = article.get('body', '').lower()
        shingles = get_shingles(text)
        m = create_minhash(shingles)
        lsh_unique.insert(idx, m)
        unique_minhash_dict[idx] = m

    # Save the MinHashLSH index and MinHashes to disk
    with open('lsh_unique.pickle', 'wb') as f:
        pickle.dump(lsh_unique, f)

    with open('minhashes_unique.pickle', 'wb') as f:
        pickle.dump(unique_minhash_dict, f)

    print("MinHashLSH index and MinHashes of unique articles have been saved.")

if __name__ == '__main__':
    main()
