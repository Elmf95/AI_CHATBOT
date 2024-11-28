import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm

# Configuration
CHUNKS_FOLDER = r"C:\Users\USER\Documents\text_chunks"
FAISS_INDEX_PATH = r"C:\Users\USER\Documents\faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64

# Charger le modèle d'embedding
print("Chargement du modèle d'embedding...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def vectorize_and_save_faiss(chunks_folder, faiss_index_path, batch_size):
    """
    Vectorise les chunks et les sauvegarde dans un index FAISS.
    """
    vector_store = None
    files = [f for f in os.listdir(chunks_folder) if f.endswith(".txt")]
    print(f"{len(files)} fichiers trouvés dans {chunks_folder}.")

    for i in tqdm(range(0, len(files), batch_size)):
        batch_files = files[i : i + batch_size]
        batch_texts = []

        for file_name in batch_files:
            file_path = os.path.join(chunks_folder, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                batch_texts.append(f.read())

        # Vectoriser le batch
        embeddings = embedding_model.embed_documents(batch_texts)

        if vector_store is None:
            # Créer un nouvel index FAISS
            vector_store = FAISS.from_texts(batch_texts, embedding_model)
        else:
            # Ajouter au vecteur existant
            vector_store.add_texts(batch_texts)

    # Sauvegarder l'index FAISS
    os.makedirs(faiss_index_path, exist_ok=True)
    vector_store.save_local(faiss_index_path)
    print(f"Index FAISS sauvegardé dans {faiss_index_path}.")


# Lancer la vectorisation
vectorize_and_save_faiss(CHUNKS_FOLDER, FAISS_INDEX_PATH, BATCH_SIZE)
