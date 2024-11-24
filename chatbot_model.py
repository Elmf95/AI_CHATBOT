from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from accelerate import infer_auto_device_map
import torch

# Configuration du modèle
model_name = "EleutherAI/gpt-neo-2.7B"

# Chargement du tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configuration pour quantization (8 bits)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Chargement du modèle avec accélération
device_map = "auto"  # Permet de répartir le modèle entre GPU et CPU
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map=device_map, quantization_config=bnb_config
)

# Définir les tokens spéciaux
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

if tokenizer.pad_token is None:  # Ajouter un token de padding si manquant
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))  # Redimensionner les embeddings

# Boucle conversationnelle
print("Bonjour, comment puis-je vous aider aujourd'hui ?.")
while True:
    # Entrée utilisateur
    user_input = input("Vous: ").strip()

    if user_input.lower() == "quit":
        print("Au revoir !")
        break

    if not user_input:  # Gestion des entrées vides
        print("Chatbot: Désolé, je n'ai pas compris votre message.")
        continue

    # Préparation des entrées pour le modèle
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)

    # Génération de la réponse
    outputs = model.generate(
        inputs.input_ids.to(
            model.device
        ),  # Les ID doivent être sur le même appareil que le modèle
        attention_mask=inputs.attention_mask.to(
            model.device
        ),  # Fournir le masque d'attention
        max_length=200,
        num_return_sequences=1,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        no_repeat_ngram_size=2,  # Empêche la répétition de phrases
    )

    # Décodage de la réponse
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Chatbot: {response}")
