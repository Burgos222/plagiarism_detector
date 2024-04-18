import os
import re
import nltk
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
"""

def preprocess_text(text):
    """
    preproccess_text es una función de preprocesamiento de texto. 
    
        Entrada: archivo de texto abierto 
        Salida: lista con los tokens del archivo

    En esta  función, se estandarizan todas las palabras a minusculas, 
    se tokenizan las palabras de cada texto, se elimian todos los stop_words y
    se hace un lemmatizer para reducir las palabras a su raíz.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def read_files(folder):
    """
    read_files es una función de lectura de archivos dentro de las 
    carpetas de prueba y de base de datos.

        Entrada: ruta de carpeta
        Salida: tupla con nombre del archivo y contenido
    
    En esta función, se recibe la ruta de un folder y comienza a iterar
    sobre los archivos del mismo para almacenarlos en una lista llamada files. 
    """
    files = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
                files.append((file_name, content))
    return files

def calculate_similarity(entry_text, database_text, vectorizer):
    """
    calculate_similarity es una función para calcular similitud de coseno
    entre archivos de entra (prueba) y archivos de la base de datos.

        Entrada: texto de prueba, texto de base de datos, matriz TF-IDF vectorizada
        Salida: float del resultado de calculo

    En esta función, se recibe la matriz vectorizada del calculo TF-IDF,
    para después calcular similitud de coseno entre archivos de prueba y archivos de base de 
    datos y regresar un valor flotante. 
    """
    entry_tfidf = vectorizer.transform([entry_text])
    database_tfidf = vectorizer.transform([database_text])
    return cosine_similarity(entry_tfidf, database_tfidf)[0][0]


def find_plagiarism(entry_files, database_files, vectorizer, threshold):
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
    plagiarism_results = []

    for entry_filename, entry_text in entry_files:
        for database_filename, database_text in database_files:
            similarity = calculate_similarity(entry_text, database_text, vectorizer)
            if similarity > threshold:
                plagiarism_results.append({
                    'entry_filename': entry_filename,
                    'database_filename': database_filename,
                    'entry_text': entry_text,
                    'database_text': database_text,
                    'similarity': similarity
                })
    return plagiarism_results

def print_plagiarism_results(plagiarism_results, threshold, vectorizer):
    """
    print_plagiarism_result es una función que imprime los casos de plagio
        
        Entrada: diccionario de resutlados de plagio, umbral y matriz TF-IDF
        Salida: 0
    
    Esta función recibe los casos de plagio los imprime y manda a llamar una función 
    que comparará los textos de ambos documentos e imprimirá los textos plagiados. 
    """
    for result in plagiarism_results:
        if result['entry_filename'] != result['database_filename']:
            print(f"\nArchivo Prueba '{result['entry_filename']}' tiene similitud del {result['similarity']:.2f}% con el archivo '{result['database_filename']}':")
            entry_text = result['entry_text']
            database_text = result['database_text']
            print_matched_lines(entry_text, database_text, threshold, vectorizer)
    
    return 0


def print_matched_lines(entry_text, database_text, threshold, vectorizer):
    """
    print_matched_lines es una función que calcula la similitud entre los textos
        
        Entrada: texto de prueba, texto de base de datos, umbral y matriz TF-IDF
        Salida: 0

    Esta función analiza los textos y calcula la similitud e imprime los textos de
    ambos archivos cuyo plagio fue encontrado.  
    """
    entry_lines = entry_text.split('\n')
    database_lines = database_text.split('\n')
    matched_lines = []
    for entry_line in entry_lines:
        for database_line in database_lines:
            similarity = calculate_similarity(entry_line, database_line, vectorizer)
            if similarity > threshold:
                matched_lines.append((entry_line, database_line))

    if matched_lines:
        print(f"Texto Plagiado: ")
        for entry_line, database_line in matched_lines:
            print(f"\nArchivo Prueba: {entry_line}")
            print(f"\nBase de Datos: {database_line}")
    else:
        print(f"No se encontró similitud con ningún archivo que sobrepase el umbral de plagio de: {threshold}.")

    return 0

def main():
    """
    Función main y controlador de parámetros
    """
    # Lectura de archivos
    entry_files = read_files('AE')
    database_files = read_files('AP')

    # Procesamiento de texto
    for i, (filename, text) in enumerate(entry_files):
        entry_files[i] = (filename, preprocess_text(text))
    for i, (filename, text) in enumerate(database_files):
        database_files[i] = (filename, preprocess_text(text))

    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer()
    all_texts = [text for _, text in entry_files + database_files]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Umbral de Similitud
    threshold = 0.2

    # Encontrar Plagio
    plagiarism_results = find_plagiarism(entry_files, database_files, vectorizer, threshold)

    # Imprimir Plagio
    print_plagiarism_results(plagiarism_results, threshold, vectorizer)

if __name__ == "__main__":
    main()