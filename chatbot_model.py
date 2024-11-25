from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Charger le modèle et le tokenizer
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# Résolution du problème de pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Charger le modèle en pleine précision ou 16 bits si un GPU est disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)


# Fonction pour générer une réponse
def generate_response(prompt, max_new_tokens=50):
    # Tokeniser l'entrée avec attention_mask explicite
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    # Générer une réponse avec contrôle des répétitions
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=0.5,
        top_p=0.8,
        do_sample=True,
        repetition_penalty=1.2,  # Réduit les répétitions
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Décoder et renvoyer le texte généré
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()


def chatbot():
    print("Bonjour, comment puis-je vous aider aujourd'hui ?")

    while True:
        user_input = input("Vous: ")
        if user_input.lower() in ["quit", "exit", "stop"]:
            print("Chatbot: Merci pour cette conversation. À bientôt !")
            break

        # Construire un prompt simple et efficace
        prompt = f"{user_input}\nRéponse :"

        # Générer et afficher la réponse
        response = generate_response(prompt)
        print(f"Chatbot: {response}")


if __name__ == "__main__":
    chatbot()
