from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from llama_cpp import Llama

# Chemins vers les fichiers et modèles
MODEL_PATH = "C:/models/kunoichi-7b.Q8_0.gguf"
FAISS_INDEX_PATH = r"C:\Users\USER\Documents\faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chargement du modèle d'embedding
print("Chargement du modèle d'embedding en cours...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Chargement de l'index FAISS
print("Chargement de l'index FAISS...")
vector_store = FAISS.load_local(
    FAISS_INDEX_PATH,
    embedding_model,
    allow_dangerous_deserialization=True,
)
print("Index FAISS chargé avec succès.")

# Chargement du modèle Llama
print("Chargement du modèle Kunoichi-7B en cours...")
llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=-1)
print("Modèle Kunoichi-7B chargé avec succès.")


# Fonction pour troncation dynamique du contexte
def truncate_context(docs, max_tokens=1000):
    """
    Tronque les documents pour ne pas dépasser une limite de tokens.
    """
    context = ""
    total_tokens = 0

    for doc in docs:
        doc_tokens = len(doc.page_content.split())
        if total_tokens + doc_tokens > max_tokens:
            break
        context += (
            f"[Source: {doc.metadata.get('source', 'inconnue')}]\n{doc.page_content}\n"
        )
        total_tokens += doc_tokens

    return context


# Fonction pour troncation dynamique de l'historique
def truncate_history(conversation, max_tokens=4096, used_tokens=0):
    """
    Tronque l'historique pour ne pas dépasser une limite de tokens.
    Garde les messages récents en priorité.
    """
    available_tokens = max_tokens - used_tokens
    truncated_history = []
    total_tokens = 0

    for user_msg, bot_msg in reversed(conversation):
        user_tokens = len(user_msg.split())
        bot_tokens = len(bot_msg.split())
        if total_tokens + user_tokens + bot_tokens > available_tokens:
            break
        truncated_history.insert(0, (user_msg, bot_msg))
        total_tokens += user_tokens + bot_tokens

    return truncated_history


# Fonction principale de chat interactif
def interactive_chat():
    print("Bienvenue dans le chatbot assistant juridique powered par ELMF.")
    print("Tapez 'quit' pour quitter la conversation.\n")

    conversation_history = []

    while True:
        user_input = input("Vous: ")
        if user_input.lower() in {"quit", "exit"}:
            print("Fin de la conversation. À bientôt !")
            break

        # Recherche des documents pertinents dans FAISS
        print("Recherche des documents pertinents...")
        k = 5
        docs = vector_store.similarity_search(user_input, k=k)
        print(f"{len(docs)} documents pertinents trouvés.")

        # Construction du contexte
        print("Construction du contexte tronqué...")
        context = truncate_context(docs)
        print("Contexte chargé.")

        # Troncation de l'historique si nécessaire
        print("Troncation de l'historique...")
        conversation_history = truncate_history(
            conversation_history, used_tokens=len(context.split())
        )
        history_text = "\n".join(
            [
                f"Utilisateur : {pair[0]}\nIA : {pair[1]}"
                for pair in conversation_history
            ]
        )

        # Construire le prompt
        print("Construction du prompt...")
        prompt = (
            f"Contexte pertinent extrait des documents :\n{context}\n\n"
            f"Historique de la conversation (récent) :\n{history_text}\n\n"
            f"Utilisateur : {user_input}\n"
            f"IA :"
        )

        # Génération de la réponse
        try:
            print("Génération de la réponse en cours...")
            response = llm(
                prompt,
                max_tokens=1000,
                temperature=0.4,
                top_k=50,
                top_p=0.9,
                stop=["Utilisateur :"],
            )["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Erreur lors de la génération de la réponse : {e}")
            continue

        print(f"IA : {response}")

        conversation_history.append((user_input, response))


if __name__ == "__main__":
    interactive_chat()
