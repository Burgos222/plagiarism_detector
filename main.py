import os
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""
    DETECTOR DE PLAGIO NLP utilizando como tecnica principal TF-IDF para similitud de texto.
    Autores: 
        * Alan Said Martinez Guzmán, A01746210
        * Alvaro Enrique Garcia Espinosa, A01781511
        * Sebastián Burgos Alanís, A01746459
    Materia y Grupo: Desarrollo de aplicaciones avanzadas de ciencias computacionales, TC3002B.201
    Fecha de entrega: Domingo 21 de abril del 2024
    
    Todas las variables están declaradas bajo el estandar snake_case
"""

"""
    preproccess_text es una función de preprocesamiento de texto. 
    
        Entrada: archivo de texto abierto 
        Salida: lista con los tokens del archivo

    En esta  función, se estandarizan todas las palabras a minusculas, 
    se tokenizan las palabras de cada texto, se elimian todos los stop_words y
    se hace un lemmatizer para reducir las palabras a su raíz.
    """

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token.strip() for token in tokens if token.strip()]
    text = ' '.join(tokens)
    return text

"""
    read_files es una función de lectura de archivos dentro de las 
    carpetas de prueba y de base de datos.

        Entrada: ruta de carpeta
        Salida: tupla con nombre del archivo y contenido
    
    En esta función, se recibe la ruta de un folder y comienza a iterar
    sobre los archivos del mismo para almacenarlos en una lista llamada files. 
    """

def read_files(folder):
    folder_path = os.path.join('.', folder)
    files = []
    for i in os.listdir(folder_path):
        filepath = os.path.join(folder_path, i)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                files.append((i, preprocess_text(file.read())))
    return files

"""
    find_plagiarism es una función que encuentra plagio entre todos los archvios de entrada y 
    base de datos.

        Entrada: archivos de prueba, archivos de la base de datos, matriz TF-IDF y umbral de plagio
        Salida: diccionario con casos de plagio

    Esta función itera todas las posibles combinaciones entre archivos de entrada y archivos de 
    base de datos, para poder analizar cada uno. Compara la similitud entre ambos y en caso de que 
    sea mayor al umbral threshold, lo guarda como un resultado de plagio, alistando el nombre 
    del archivo de entrada, el archivo de base de datos con el que tuvo coincidencia, sus textos, 
    y su similitud en porcentaje. 
    """

def detect_plagiarism(entry_files, database_files, vectorizer, threshold):
    if not isinstance(entry_files, list) or not isinstance(database_files, list):
        raise TypeError("entry_files y database_files deben ser listas")

    if not entry_files or not database_files:
        raise ValueError("entry_files y database_files no pueden estar vacías")

    for file_list in [entry_files, database_files]:
        for file_tuple in file_list:
            if not isinstance(file_tuple, tuple) or len(file_tuple) != 2:
                raise ValueError("Cada elemento de la lista debe ser una tupla de dos strings")
            if not all(isinstance(item, str) for item in file_tuple):
                raise ValueError("Cada elemento de la tupla debe ser un string")
              
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError("El vectorizer debe ser un objeto de tipo TfidfVectorizer")

    plagiarism_results = []

    for entry_filename, entry_text in entry_files:
        entry_tfidf = vectorizer.transform([entry_text])

        for database_filename, database_text in database_files:
            database_tfidf = vectorizer.transform([database_text])

            similarity = cosine_similarity(entry_tfidf, database_tfidf)[0][0]
            if similarity > threshold:
                plagiarism_report = {
                    'entry_filename': entry_filename,
                    'database_filename': database_filename,
                    'similarity': similarity,
                    'plagiarism_report': ''
                }

                entry_lines = entry_text.split('\n')
                database_lines = database_text.split('\n')
                matched_lines = []

                for entry_line, database_line in zip(entry_lines, database_lines):
                    entry_tfidf_line = vectorizer.transform([entry_line])
                    database_tfidf_line = vectorizer.transform([database_line])
                    line_similarity = cosine_similarity(entry_tfidf_line, database_tfidf_line)[0][0]
                    if line_similarity > threshold:
                        matched_lines.append((entry_line, database_line))

                if matched_lines:
                    plagiarism_report_text = "Texto Plagiado:\n"
                    for entry_line, database_line in matched_lines:
                        entry_words = entry_line.split()
                        database_words = database_line.split()
                        matched_words = [(entry_word, database_word) for entry_word, database_word in zip(entry_words, database_words) if entry_word == database_word]
                        if matched_words:
                            for entry_word, database_word in matched_words:
                                plagiarism_report_text += f"{entry_word} "
                else:
                    plagiarism_report_text = f"No se encontró similitud con ningún archivo que sobrepase el umbral de plagio de: {threshold}."

                plagiarism_report['plagiarism_report'] = plagiarism_report_text
                plagiarism_results.append(plagiarism_report)

    return plagiarism_results

"""
    Zona de control de parámetros
"""

entry_files = read_files('AP')
database_files = read_files('AS')
vectorizer = TfidfVectorizer()
all_texts = [text for _, text in entry_files + database_files]
tfidf_matrix = vectorizer.fit_transform(all_texts)
threshold = 0.6

results = detect_plagiarism(entry_files, database_files, vectorizer, threshold)

for result in results:
    print(f"\nArchivo Prueba '{result['entry_filename']}' tiene similitud del {result['similarity']:.2f}% con el archivo '{result['database_filename']}':")
    print(result['plagiarism_report'])