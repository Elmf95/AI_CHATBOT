import pdfplumber
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor

# Configuration
PDF_PATH = r"C:\Users\USER\Documents\codes.pdf"
OUTPUT_FOLDER = r"C:\Users\USER\Documents\text_chunks"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

BATCH_SIZE = 500
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_WORKERS = 8


def extract_pages(pdf_path, start, end):
    """
    Extrait les pages d'un fichier PDF dans une plage donnée.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page_num in range(start, end):
                text += pdf.pages[page_num].extract_text() + "\n"
            print(f"Pages {start + 1} à {end} extraites.")
            return text
    except Exception as e:
        print(f"Erreur lors de l'extraction des pages {start + 1} à {end}: {e}")
        return ""


def split_and_save_text(batch_text, batch_index):
    """
    Divise le texte en chunks et les sauvegarde en fichiers séparés.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(batch_text)
    for i, chunk in enumerate(chunks):
        file_name = os.path.join(OUTPUT_FOLDER, f"batch_{batch_index}_chunk_{i+1}.txt")
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(chunk)
    print(f"Batch {batch_index} découpé et sauvegardé ({len(chunks)} chunks).")


def process_batch(pdf_path, start, end, batch_index):
    """
    Traite un lot : extrait les pages et divise en chunks.
    """
    batch_text = extract_pages(pdf_path, start, end)
    split_and_save_text(batch_text, batch_index)


def parallel_extraction_and_split(pdf_path, batch_size, max_workers):
    """
    Effectue l'extraction et le découpage en parallèle.
    """
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Le fichier contient {total_pages} pages.")

        # Préparer les plages de pages pour chaque lot
        page_ranges = [
            (i, min(i + batch_size, total_pages))
            for i in range(0, total_pages, batch_size)
        ]

        # Lancer le traitement en parallèle
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_batch, pdf_path, start, end, index)
                for index, (start, end) in enumerate(page_ranges)
            ]
            for future in futures:
                future.result()  # Attendre la fin de chaque tâche


# Lancer le traitement
parallel_extraction_and_split(PDF_PATH, BATCH_SIZE, MAX_WORKERS)
