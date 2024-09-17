import os
from datasketch import MinHash, MinHashLSH
from collections import defaultdict

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def create_minhash(text):
    minhash = MinHash()
    for d in text.split():
        minhash.update(d.encode('utf-8'))
    return minhash

def find_near_duplicates(directory):
    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    file_minhashes = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                content = read_file(file_path)
                minhash = create_minhash(content)
                file_minhashes[file_path] = minhash
                lsh.insert(file_path, minhash)
    
    near_duplicates = defaultdict(list)
    for file_path, minhash in file_minhashes.items():
        result = lsh.query(minhash)
        if len(result) > 1:
            near_duplicates[os.path.dirname(file_path)].append((file_path, result))
    
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
                if similar_file != file_path:
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