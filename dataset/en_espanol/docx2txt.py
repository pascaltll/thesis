from docx import Document
import os

def docx_to_txt(docx_filename):
    doc = Document(docx_filename)
    txt_filename = os.path.splitext(docx_filename)[0] + ".txt"
    
    with open(txt_filename, "w", encoding="utf-8") as txt_file:
        for para in doc.paragraphs:
            txt_file.write(para.text + "\n")
    
    print(f"Archivo convertido: {txt_filename}")

# Lista de archivos DOCX en el directorio actual
docx_files = [f for f in os.listdir('.') if f.endswith(".docx")]

# Verificar que hay al menos dos archivos DOCX
docx_to_txt(docx_files[0])
docx_to_txt(docx_files[1])


