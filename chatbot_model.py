from llama_cpp import Llama

# Remplacez par le chemin de votre modèle GGUF
MODEL_PATH = "C:/models/kunoichi-7b.Q8_0.gguf"


def interactive_chat():
    """
    Lance une boucle interactive de discussion avec le modèle GGUF.
    """
    print("Chargement du modèle... Cela peut prendre un moment.")
    llm = Llama(model_path=MODEL_PATH)
    print("Modèle chargé avec succès ! Vous pouvez commencer à discuter.")
    print("Tapez 'quit' pour quitter la conversation.\n")

    conversation_context = ""  # Qui garde le contexte de la discussion

    while True:
        user_input = input("Vous: ")
        if user_input.lower() in {"quit", "exit"}:
            print("Fin de la conversation. À bientôt !")
            break

        # Construire le prompt en ajoutant le contexte de la conversation
        prompt = conversation_context + f"User: {user_input}\nAI:"
        response = llm(prompt, max_tokens=200, stop=["User:"])["choices"][0][
            "text"
        ].strip()

        # Affiche la réponse et met à jour le contexte
        print(f"Modèle: {response}")
        conversation_context += f"User: {user_input}\nAI: {response}\n"


# Lancer la discussion
if __name__ == "__main__":
    interactive_chat()
