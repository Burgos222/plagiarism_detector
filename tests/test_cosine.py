import unittest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from main_2 import calculate_similarity

class TestCalculateSimilarity(unittest.TestCase):
    """
    A test case class for the calculate_similarity function.
    """

    def setUp(self):
        """ Configurar vectorizador de pruebas"""
        self.vectorizer = TfidfVectorizer()

        # Datos de ejemplo para la prueba
        self.entry_text = "This is a test entry text."
        self.database_text = "This is a database text."
        
        # Ajustar el vectorizador con los textos de entrada y base de datos
        self.vectorizer.fit([self.entry_text, self.database_text])

    def test_similarity(self):
        """Probar la consistencia de los resultados de la función calculate_similarity."""
        expected_similarity = cosine_similarity(self.vectorizer.transform([self.entry_text]), 
                                                self.vectorizer.transform([self.database_text]))[0][0]
        similarity = calculate_similarity(self.entry_text, self.database_text, self.vectorizer)
        
        self.assertAlmostEqual(similarity, expected_similarity, places=5)

