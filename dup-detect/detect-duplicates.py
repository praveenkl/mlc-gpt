import os
from datasketch import MinHash, MinHashLSH

# def tokenize(text, k=5):
#     """Tokenize text into k-word shingles."""
#     words = text.split()
#     return [' '.join(words[i:i+k]) for i in range(len(words) - k + 1)]

def create_minhash(text):
    minhash = MinHash()
    for d in text.split():
        minhash.update(d.encode('utf-8'))
    return minhash

def find_near_duplicates_in_directory(directory_path, threshold=0.5, num_perm=128):
    """Find near-duplicate text files within a directory."""
    text_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    minhashes = {}
    
    # Compute MinHash for each text file
    for text_file in text_files:
        file_path = os.path.join(directory_path, text_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        minhashes[text_file] = create_minhash(text)
    
    # Initialize LSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for filename, minhash in minhashes.items():
        lsh.insert(filename, minhash)
    
    # Find near-duplicates
    near_duplicates = set()
    for filename, minhash in minhashes.items():
        similar_files = lsh.query(minhash)
        for similar_file in similar_files:
            if similar_file != filename:
                pair = tuple(sorted([filename, similar_file]))
                near_duplicates.add(pair)
    
    return near_duplicates

def generate_report(main_directory):
    """Generate a report of near-duplicate text files in each sub-directory."""
    sub_dirs = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
    
    for sub_dir in sub_dirs:
        print(f'Processing sub-directory: {sub_dir}')
        sub_dir_path = os.path.join(main_directory, sub_dir)
        near_duplicates = find_near_duplicates_in_directory(sub_dir_path)
        
        if near_duplicates:
            print(f'Near-duplicates in {sub_dir}:')
            for file1, file2 in near_duplicates:
                print(f'\t{file1} and {file2}')
        else:
            print(f'No near-duplicates found in {sub_dir}')

# Specify the path to your main directory
main_dir = 'queries'

# Generate the report
generate_report(main_dir)
