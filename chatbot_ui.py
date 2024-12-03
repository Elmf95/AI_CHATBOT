import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from llama_cpp import Llama

# Configuration initiale
MODEL_PATH = "C:/models/kunoichi-7b.Q8_0.gguf"
FAISS_INDEX_PATH = r"C:\Users\USER\Documents\faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# Chargement des modèles et de l'index
@st.cache_resource
def load_models():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True
    )

    llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=-1)

    return embedding_model, vector_store, llm


embedding_model, vector_store, llm = load_models()


# Fonctions utilitaires
def truncate_context(docs, max_tokens=1000):
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


def truncate_history(conversation, max_tokens=4096, used_tokens=0):
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


# Interface utilisateur Streamlit
st.title(" Assistant Juridique powered by ELMF")
st.write(
    "Posez vos questions et obtenez une réponse basée sur l'ensemble des lois françaises."
)

# Si l'historique n'existe pas dans la session, l'initialiser
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

user_input = st.text_input("Votre question :", key="user_input")
if st.button("Envoyer"):
    if user_input:
        st.write("Recherche des documents pertinents...")
        docs = vector_store.similarity_search(user_input, k=5)

        st.write("Construction du contexte...")
        context = truncate_context(docs)

        # Troncation de l'historique

        conversation_history = truncate_history(
            st.session_state["conversation_history"], used_tokens=len(context.split())
        )
        history_text = "\n".join(
            [
                f"Utilisateur : {pair[0]}\nIA : {pair[1]}"
                for pair in conversation_history
            ]
        )

        # Construire le prompt
        prompt = (
            f"Contexte pertinent extrait des documents :\n{context}\n\n"
            f"Historique de la conversation (récent) :\n{history_text}\n\n"
            f"Utilisateur : {user_input}\nIA :"
        )

        # Génération de la réponse
        try:
            response = llm(
                prompt,
                max_tokens=1000,
                temperature=0.4,
                top_k=50,
                top_p=0.9,
                stop=["Utilisateur :"],
            )["choices"][0]["text"].strip()
        except Exception as e:
            response = f"Erreur lors de la génération de la réponse : {e}"

        st.write(f"**Réponse du chatbot :** {response}")

        # Ajouter la question et la réponse à l'historique
        st.session_state["conversation_history"].append((user_input, response))

        # Afficher l'historique de la conversation
        st.write("**Historique de la conversation**:")
        for user_msg, bot_msg in st.session_state["conversation_history"]:
            st.write(f"Utilisateur : {user_msg}")
            st.write(f"IA : {bot_msg}")
