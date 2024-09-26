import os
import json

# Replace 'path/to/directory' with the actual path to your directory
directory = './unique_articles'

total_articles = 0
articles_with_duplicates = 0
total_duplicate_counts = 0

# List to keep track of articles and their duplicate counts
duplicate_counts_list = []

for filename in os.listdir(directory):
    if filename.endswith('.json'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {filename}: {e}")
                continue

            for article in data:
                total_articles += 1
                duplicates = article.get('duplicate_uris', [])
                num_duplicates = len(duplicates)

                if num_duplicates > 0:
                    articles_with_duplicates += 1
                    total_duplicate_counts += num_duplicates

                    # Add article's URI and duplicate count to the list
                    article_uri = article.get('uri', 'Unknown')
                    duplicate_counts_list.append((article_uri, num_duplicates))

# Calculate average number of duplicate URIs for articles with duplicates
if articles_with_duplicates > 0:
    average_duplicate_uris = total_duplicate_counts / articles_with_duplicates
else:
    average_duplicate_uris = 0

# Sort the list of articles by duplicate count in descending order
duplicate_counts_list.sort(key=lambda x: x[1], reverse=True)

# Get the top 5 articles
top_5_articles = duplicate_counts_list[:5]

print("Total number of articles:", total_articles)
print("Number of articles with one or more elements in 'duplicate_uris':", articles_with_duplicates)
print("URIs of the top 5 articles with the most duplicate_uris:")
for uri, count in top_5_articles:
    print(f"URI: {uri}, Number of duplicate_uris: {count}")
print("Average number of duplicate_uris for articles with duplicates:", average_duplicate_uris)
