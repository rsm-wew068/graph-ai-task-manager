from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Global variable to hold the model (lazy loaded)
model = None

def get_model():
    """Get the sentence transformer model, loading it if necessary."""
    global model
    if model is None:
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Warning: Failed to load sentence transformer model: {e}")
            # Fallback: return None and handle gracefully in calling functions
            return None
    return model

def chunk_text(text, chunk_size=512):
    tokens = text.split()
    return [' '.join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]

def build_faiss_index(chunks, batch_size=512, max_chunks=100000):
    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]

    model = get_model()
    if model is None:
        raise RuntimeError("Failed to load sentence transformer model")
        
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dim)

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        batch = chunks[start:end]
        vectors = model.encode(batch, show_progress_bar=False)
        index.add(np.array(vectors, dtype=np.float32))

    return index, chunks

def save_index(index, path="faiss_index.idx"):
    faiss.write_index(index, path)

def load_index(path="faiss_index.idx"):
    return faiss.read_index(path)

def embed_dataframe(df, content_col="content"):
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    df_copy["chunks"] = df_copy[content_col].apply(lambda x: chunk_text(x))
    all_chunks = df_copy["chunks"].explode().dropna().tolist()
    index, chunks = build_faiss_index(all_chunks)
    save_index(index)
    return index, chunks


def retrieve_similar_chunks(query, index=None, chunks=None, k=3):
    if index is None:
        index = load_index()
    if chunks is None:
        raise ValueError("Must provide chunks list for index reference.")

    model = get_model()
    if model is None:
        raise RuntimeError("Failed to load sentence transformer model")
        
    vector = model.encode([query])
    D, I = index.search(np.array(vector, dtype=np.float32), k)
    return [chunks[i] for i in I[0]]
