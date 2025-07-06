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


def embed_dataframe(df, content_col="content"):
    """
    Create embeddings for dataframe content without saving to disk.
    Returns index and chunks for in-memory use only.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    df_copy["chunks"] = df_copy[content_col].apply(lambda x: chunk_text(x))
    all_chunks = df_copy["chunks"].explode().dropna().tolist()
    
    # Build index but don't save to disk
    index, chunks = build_faiss_index(all_chunks)
    
    return index, chunks


def retrieve_similar_chunks(query, index, chunks, k=3):
    """
    Retrieve similar chunks using provided index and chunks.
    No file loading required - works with in-memory objects.
    """
    if index is None or chunks is None:
        raise ValueError(
            "Must provide both index and chunks for similarity search."
        )

    model = get_model()
    if model is None:
        raise RuntimeError("Failed to load sentence transformer model")
        
    vector = model.encode([query])
    distances, indices = index.search(np.array(vector, dtype=np.float32), k)
    
    # Handle case where index might be smaller than k
    valid_indices = [i for i in indices[0] if i < len(chunks)]
    return [chunks[i] for i in valid_indices]


def embed_email_rows(email_df, content_col="content"):
    """
    Create embeddings for email DataFrame with all metadata preserved.
    Returns index, chunks, and row mapping for in-memory use only.
    
    Args:
        email_df: DataFrame with all email columns
        content_col: Column name containing email content for embedding
        
    Returns:
        tuple: (index, chunks, email_rows) where email_rows maps chunks to rows
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = email_df.copy()
    
    # Create chunks but keep track of which email they came from
    df_copy["chunks"] = df_copy[content_col].apply(lambda x: chunk_text(x))
    
    # Create mapping from chunk index to email row
    email_rows = []
    all_chunks = []
    
    for idx, row in df_copy.iterrows():
        chunks = row["chunks"]
        for chunk in chunks:
            if chunk and chunk.strip():  # Only add non-empty chunks
                all_chunks.append(chunk)
                email_rows.append(row.to_dict())
    
    # Build index but don't save to disk
    if all_chunks:
        index, _ = build_faiss_index(all_chunks)
        return index, all_chunks, email_rows
    else:
        return None, [], []
