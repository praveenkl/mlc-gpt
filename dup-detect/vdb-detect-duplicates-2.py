import os
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def find_near_duplicates(directory):
    file_paths = []
    file_contents = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                file_contents.append(read_file(file_path))
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(file_contents)
    
    # Convert sparse matrix to dense
    dense_vectors = tfidf_matrix.toarray().astype('float32')
    
    # Normalize the vectors
    faiss.normalize_L2(dense_vectors)
    
    # Create FAISS index
    dimension = dense_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(dense_vectors)
    
    # Search for similar vectors
    k = 5  # number of nearest neighbors to retrieve
    similarity_threshold = 0.8  # adjust this value to control the similarity threshold
    
    _, I = index.search(dense_vectors, k)
    
    near_duplicates = defaultdict(list)
    for i, neighbors in enumerate(I):
        similar_files = []
        for j in neighbors[1:]:  # Skip the first one as it's the file itself
            if np.dot(dense_vectors[i], dense_vectors[j]) > similarity_threshold:
                similar_files.append(file_paths[j])
        if similar_files:
            near_duplicates[os.path.dirname(file_paths[i])].append((file_paths[i], similar_files))
    
    return near_duplicates

def generate_report(near_duplicates):
    report = []
    for directory, duplicates in near_duplicates.items():
        report.append(f"Directory: {directory}")
        report.append(f"Number of files with near-duplicates: {len(duplicates)}")
        for file_path, similar_files in duplicates:
            report.append(f"  File: {os.path.basename(file_path)}")
            report.append("  Similar files:")
            for similar_file in similar_files:
                report.append(f"    - {os.path.basename(similar_file)}")
        report.append("")
    return "\n".join(report)

def main(root_directory):
    near_duplicates = find_near_duplicates(root_directory)
    report = generate_report(near_duplicates)
    
    with open("near_duplicate_report.txt", "w") as f:
        f.write(report)
    
    print("Report generated: near_duplicate_report.txt")

if __name__ == "__main__":
    root_directory = input("Enter the path to the root directory: ")
    main(root_directory)