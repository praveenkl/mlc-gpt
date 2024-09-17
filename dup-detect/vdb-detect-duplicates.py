import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util

def find_near_duplicates_in_directory(directory_path, threshold=0.8, model_name='all-MiniLM-L6-v2'):
    """Find near-duplicate text files within a directory using a vector database."""
    text_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    embeddings = []
    filenames = []
    
    # Load the pre-trained model
    model = SentenceTransformer(model_name)
    
    # Compute embeddings for each text file
    for text_file in text_files:
        file_path = os.path.join(directory_path, text_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        embedding = model.encode(text, convert_to_tensor=False)
        embeddings.append(embedding)
        filenames.append(text_file)
    
    if not embeddings:
        return set()
    
    embeddings = np.array(embeddings).astype('float32')
    
    # Create a Faiss index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Perform similarity search
    similarities, indices = index.search(embeddings, k=len(embeddings))
    
    near_duplicates = set()
    for i, sims in enumerate(similarities):
        for j, sim in zip(indices[i], sims):
            if i != j and sim >= threshold:
                pair = tuple(sorted([filenames[i], filenames[j]]))
                near_duplicates.add(pair)
    
    return near_duplicates

def generate_report(main_directory, threshold=0.8):
    """Generate a report of near-duplicate text files in each sub-directory."""
    sub_dirs = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
    
    for sub_dir in sub_dirs:
        print(f'Processing sub-directory: {sub_dir}')
        sub_dir_path = os.path.join(main_directory, sub_dir)
        near_duplicates = find_near_duplicates_in_directory(sub_dir_path, threshold=threshold)
        
        if near_duplicates:
            print(f'Near-duplicates in {sub_dir}:')
            for file1, file2 in near_duplicates:
                print(f'\t{file1} and {file2}')
        else:
            print(f'No near-duplicates found in {sub_dir}')
    
    print('Processing completed.')

# Specify the path to your main directory
main_dir = 'queries'

# Generate the report
generate_report(main_dir, threshold=0.8)
