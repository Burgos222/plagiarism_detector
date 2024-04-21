import unittest
from main import preprocess_text

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

class TestPreprocessText(unittest.TestCase):
    """
    Casos de prueba para revisar el preprocesamiento de los archivos
    """

    """
    Caso de prueba para comprobar el lowercasing de los textos.
    """
    def test_lowercase(self):
        text = "This is A Test String"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "test string")

    """
    Caso de prrueba para comprobar la eliminación de caracteres especiales.
    """
    def test_remove_special_characters(self):
        text = "This is a test string with special characters: @#$%&*"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "test string special character")

    """
    Caso de prueba para comprobar la eliminación de stopwords.
    """
    def test_remove_stopwords(self):
        text = "This is a test string with some stopwords such as 'is', 'a', and 'with'"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "test string stopwords")

    """
    Caso de prueba para comprobar la lematización de los textos.
    """
    def test_lemmatization(self):
        text = "cats running in the fields"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "cat running field")

