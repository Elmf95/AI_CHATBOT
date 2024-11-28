from langchain_community.llms import LlamaCpp
from llama_cpp import Llama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Chemins vers les fichiers et modèles
MODEL_PATH = "C:/models/Llama-2-7B-GGUF"
FAISS_INDEX_PATH = r"C:\Users\USER\Documents\faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Charger le modèle d'embedding
print("Chargement du modèle d'embedding...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Charger l'index FAISS
print("Chargement de l'index FAISS...")
vector_store = FAISS.load_local(
    FAISS_INDEX_PATH,
    embedding_model,
    allow_dangerous_deserialization=True,  # Autoriser explicitement la désérialisation
)
print("Index FAISS chargé avec succès.")

# Charger le modèle Llama-2 avec extension de la taille du contexte
print("Chargement du modèle Llama-2-7B-GGUF...")


# Llama avec une plus grande limite de contexte
llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=-1)
print("Modèle GGUF Llama-2-7B chargé avec succès ! Vous pouvez commencer à discuter.")


def interactive_chat():
    """
    Lance une boucle interactive de discussion avec le modèle et l'index FAISS.
    """
    print("Tapez 'quit' pour quitter la conversation.\n")

    conversation_context = ""  # Qui garde le contexte de la discussion

    while True:
        user_input = input("Vous: ")
        if user_input.lower() in {"quit", "exit"}:
            print("Fin de la conversation. À bientôt !")
            break

        # Recherche dans FAISS
        docs = vector_store.similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        # Tronquer le contexte pour ne pas dépasser la limite de tokens
        # On garde un maximum de 4096 tokens, ce qui doit être ajusté en fonction des tokens par extrait
        truncated_context = context[:4096]

        # Construire le prompt avec le contexte tronqué
        prompt = f"Contexte : {truncated_context}\nUser: {user_input}\nAI:"

        # Générer la réponse à partir du modèle
        response = llm(prompt, max_tokens=100, stop=["User:"])["choices"][0][
            "text"
        ].strip()

        # Afficher la réponse et mettre à jour le contexte
        print(f"Modèle: {response}")
        conversation_context += f"User: {user_input}\nAI: {response}\n"


# Lancer la discussion
if __name__ == "__main__":
    interactive_chat()
