import os
import json
import pickle
from datasketch import MinHash, MinHashLSH
from nltk.tokenize import word_tokenize
import nltk

# Ensure the required NLTK data files are downloaded
nltk.download('punkt', quiet=True)

def read_articles(root_dir):
    articles = []
    article_ids = []
    article_id = 0
    for year in os.listdir(root_dir):
        year_path = os.path.join(root_dir, year)
        if os.path.isdir(year_path):
            for month in os.listdir(year_path):
                month_path = os.path.join(year_path, month)
                if os.path.isdir(month_path):
                    for filename in os.listdir(month_path):
                        if filename.startswith('articles_') and filename.endswith('.json'):
                            file_path = os.path.join(month_path, filename)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                try:
                                    data = json.load(f)
                                    for article in data:
                                        articles.append(article)
                                        article_ids.append(article_id)
                                        article_id += 1
                                except json.JSONDecodeError as e:
                                    print(f"Error decoding JSON from {file_path}: {e}")
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

def main():
    root_dir = 'raw'  # Adjust this if your directory is named differently
    articles, article_ids = read_articles(root_dir)
    
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
        # Sort group_indices to ensure consistent order
        group_indices.sort()
        # Retain the article with the lowest index
        retained_idx = group_indices[0]
        retained_article = articles[article_ids.index(retained_idx)]
        # Collect URIs of duplicate articles (if any)
        duplicate_uris = []
        for idx in group_indices[1:]:
            article = articles[article_ids.index(idx)]
            uri = article.get('uri')
            if uri:
                duplicate_uris.append(uri)
        if duplicate_uris:
            # Add 'duplicate_uris' field to the retained article
            retained_article['duplicate_uris'] = duplicate_uris
        unique_articles.append(retained_article)
    
    # Create output directory if it doesn't exist
    output_dir = 'unique_articles'
    os.makedirs(output_dir, exist_ok=True)
    
    # Write each unique article to a separate file
    for i, article in enumerate(unique_articles):
        output_filename = os.path.join(output_dir, f'unique_article_{i}.json')
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(article, f, ensure_ascii=False, indent=2)
        print(f"Unique article {i} written to {output_filename}")
    
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
