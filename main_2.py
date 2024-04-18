import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess_text(text):
    """Preprocess the text: lowercase, tokenization, removal of punctuation and numbers, stop words removal, lemmatization."""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def read_files(folder):
    """Read files from the specified folder and return a list of (filename, content) tuples."""
    files = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
                files.append((file_name, content))
    return files

def calculate_similarity(entry_text, database_text, vectorizer):
    """Calculate cosine similarity between entry and database texts."""
    entry_tfidf = vectorizer.transform([entry_text])
    database_tfidf = vectorizer.transform([database_text])
    return cosine_similarity(entry_tfidf, database_tfidf)[0][0]


def find_plagiarism(entry_files, database_files, vectorizer, threshold):
    """Find plagiarism between entry files and database files."""
    plagiarism_results = []

    for entry_filename, entry_text in entry_files:
        for database_filename, database_text in database_files:
            similarity = calculate_similarity(entry_text, database_text, vectorizer)
            if similarity > threshold:
                plagiarism_results.append({
                    'entry_filename': entry_filename,
                    'database_filename': database_filename,
                    'entry_text': entry_text,  # Include entry text
                    'database_text': database_text,  # Include database text
                    'similarity': similarity
                })
    return plagiarism_results

def print_plagiarism_results(plagiarism_results, threshold, vectorizer):
    """Print plagiarism results."""
    for result in plagiarism_results:
        if result['entry_filename'] != result['database_filename']:
            print(f"\nArchivo Prueba '{result['entry_filename']}' tiene similitud del {result['similarity']:.2f}% con el archivo '{result['database_filename']}':")
            entry_text = result['entry_text']
            database_text = result['database_text']
            print_matched_lines(entry_text, database_text, threshold, vectorizer)


def print_matched_lines(entry_text, database_text, threshold, vectorizer):
    """Print matched lines between entry and database texts."""
    entry_lines = entry_text.split('\n')
    database_lines = database_text.split('\n')
    matched_lines = []
    for entry_line, database_line in zip(entry_lines, database_lines):
        similarity = calculate_similarity(entry_line, database_line, vectorizer)
        if similarity > threshold:
            matched_lines.append((entry_line, database_line))

    if matched_lines:
        print("Texto Plagiado:")
        for entry_line, database_line in matched_lines:
            print(f"Archivo Prueba: {entry_line}")
            print(f"Base de Datos: {database_line}")
    else:
        print(f"No se encontró similitud con ningún archivo que sobrepase el umbral de plagio de: {threshold}.")

def main():
    # Read files
    entry_files = read_files('AE')
    database_files = read_files('AP')

    # Preprocess text
    for i, (filename, text) in enumerate(entry_files):
        entry_files[i] = (filename, preprocess_text(text))
    for i, (filename, text) in enumerate(database_files):
        database_files[i] = (filename, preprocess_text(text))

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    all_texts = [text for _, text in entry_files + database_files]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Set similarity threshold
    threshold = 0.2

    # Find plagiarism
    plagiarism_results = find_plagiarism(entry_files, database_files, vectorizer, threshold)

    # Print plagiarism results
    print_plagiarism_results(plagiarism_results, threshold, vectorizer)

if __name__ == "__main__":
    main()