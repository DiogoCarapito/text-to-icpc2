import txtai
import pandas as pd


def main():
    embeddings = txtai.Embeddings({
        'path': 'sentence-transformers/all-MiniLM-L6-v2',
    })
    
    df = pd.read_csv('data/icpc2_processed.csv')
    

    embeddings.index(df["index_seach"].dropna().values)
    embeddings.save('embeddings.tar.gz')
    
    print("Embeddings saved")

if __name__ == "__main__":
    main()