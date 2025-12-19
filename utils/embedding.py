import numpy as np
import faiss
import os
from typing import List, Union
from openai import OpenAI

# Handle dotenv import gracefully for deployment environments
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Global variable to hold the OpenAI client (lazy loaded)
openai_client = None


def get_openai_client():
    """Get the OpenAI client, loading it if necessary."""
    global openai_client
    if openai_client is None:
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            openai_client = OpenAI(api_key=api_key)
            print("✅ OpenAI embedding client initialized")
        except Exception as e:
            print(f"❌ Warning: Failed to initialize OpenAI client: {e}")
            return None
    return openai_client


def get_openai_embeddings(
    texts: List[str], model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """Get embeddings from OpenAI API for a list of texts."""
    client = get_openai_client()
    if client is None:
        raise RuntimeError("Failed to initialize OpenAI client")

    try:
        # Clean and validate input texts
        cleaned_texts = []
        for text in texts:
            if isinstance(text, str) and text.strip():
                # Truncate very long texts to avoid API limits (8191 tokens for text-embedding-3-small)
                cleaned_text = text.strip()[:8000]  # Conservative limit
                cleaned_texts.append(cleaned_text)
            else:
                cleaned_texts.append("empty")  # Fallback for empty/invalid texts

        response = client.embeddings.create(input=cleaned_texts, model=model)

        embeddings = [item.embedding for item in response.data]
        return embeddings
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        raise RuntimeError(f"Failed to get OpenAI embeddings: {e}")


def get_embedding_dimension(model: str = "text-embedding-3-small") -> int:
    """Get the embedding dimension for the specified model."""
    model_dimensions = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    return model_dimensions.get(model, 1536)


def chunk_text(text, chunk_size=512):
    tokens = text.split()
    return [
        " ".join(tokens[i : i + chunk_size]) for i in range(0, len(tokens), chunk_size)
    ]


def build_faiss_index(
    chunks, batch_size=100, max_chunks=100000, embedding_model="text-embedding-3-small"
):
    """Build FAISS index using OpenAI embeddings."""
    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
        print(f"⚠️ Truncated to {max_chunks} chunks for embedding")

    client = get_openai_client()
    if client is None:
        raise RuntimeError("Failed to initialize OpenAI client for embeddings")

    dim = get_embedding_dimension(embedding_model)
    index = faiss.IndexFlatL2(dim)

    print(f"🔧 Building FAISS index with OpenAI {embedding_model} (dim={dim})")
    print(f"📊 Processing {len(chunks)} chunks in batches of {batch_size}")

    # Process in smaller batches for OpenAI API
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch = chunks[start:end]

        try:
            # Get embeddings from OpenAI
            embeddings = get_openai_embeddings(batch, embedding_model)
            vectors = np.array(embeddings, dtype=np.float32)
            index.add(vectors)

            if start % (batch_size * 5) == 0:  # Progress every 5 batches
                print(f"📈 Processed {end}/{len(chunks)} chunks...")

        except Exception as e:
            print(f"❌ Error processing batch {start}-{end}: {e}")
            # Continue with next batch rather than failing completely
            continue

    print(f"✅ FAISS index built with {index.ntotal} vectors")
    return index, chunks


def embed_dataframe(
    df, content_col="content", embedding_model="text-embedding-3-small"
):
    """
    Create OpenAI embeddings for dataframe content without saving to disk.
    Returns index and chunks for in-memory use only.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    df_copy["chunks"] = df_copy[content_col].apply(
        lambda x: chunk_text(x) if isinstance(x, str) else []
    )
    all_chunks = df_copy["chunks"].explode().dropna().tolist()

    print(f"📊 Creating OpenAI embeddings for {len(all_chunks)} chunks...")

    # Build index using OpenAI embeddings
    try:
        index, chunks = build_faiss_index(all_chunks, embedding_model=embedding_model)
        return index, chunks
    except Exception as e:
        print(f"❌ Failed to create dataframe embeddings: {e}")
        return None, []


def retrieve_similar_chunks(
    query, index, chunks, k=3, embedding_model="text-embedding-3-small"
):
    """
    Retrieve similar chunks using OpenAI embeddings for the query.
    No file loading required - works with in-memory objects.
    """
    if index is None or chunks is None:
        raise ValueError("Must provide both index and chunks for similarity search.")

    client = get_openai_client()
    if client is None:
        raise RuntimeError("Failed to initialize OpenAI client for query embedding")

    try:
        # Get embedding for the query using OpenAI
        query_embeddings = get_openai_embeddings([query], embedding_model)
        query_vector = np.array(query_embeddings[0], dtype=np.float32).reshape(1, -1)

        # Search the FAISS index
        distances, indices = index.search(query_vector, k)

        # Handle case where index might be smaller than k
        valid_indices = [i for i in indices[0] if i < len(chunks)]
        return [chunks[i] for i in valid_indices]

    except Exception as e:
        print(f"❌ Error during similarity search: {e}")
        raise RuntimeError(f"Failed to retrieve similar chunks: {e}")


def embed_email_rows(
    email_df, content_col="content", embedding_model="text-embedding-3-small"
):
    """
    Create OpenAI embeddings for email DataFrame with all metadata preserved.
    Returns index, chunks, and row mapping for in-memory use only.

    Args:
        email_df: DataFrame with all email columns
        content_col: Column name containing email content for embedding
        embedding_model: OpenAI embedding model to use

    Returns:
        tuple: (index, chunks, email_rows) where email_rows maps chunks to rows
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = email_df.copy()

    # Create chunks but keep track of which email they came from
    print(f"📧 Processing {len(df_copy)} emails for embedding...")
    df_copy["chunks"] = df_copy[content_col].apply(
        lambda x: chunk_text(x) if isinstance(x, str) else []
    )

    # Create mapping from chunk index to email row
    email_rows = []
    all_chunks = []

    for idx, row in df_copy.iterrows():
        chunks = row["chunks"]
        for chunk in chunks:
            if chunk and chunk.strip():  # Only add non-empty chunks
                all_chunks.append(chunk)
                email_rows.append(row.to_dict())

    print(f"📝 Created {len(all_chunks)} chunks from {len(df_copy)} emails")

    # Build index using OpenAI embeddings
    if all_chunks:
        try:
            index, _ = build_faiss_index(all_chunks, embedding_model=embedding_model)
            print(f"✅ Successfully created OpenAI embeddings index")
            return index, all_chunks, email_rows
        except Exception as e:
            print(f"❌ Failed to create embeddings: {e}")
            return None, [], []
    else:
        print("⚠️ No valid chunks found for embedding")
        return None, [], []
