import unittest
from sklearn.feature_extraction.text import TfidfVectorizer
from main_2 import find_plagiarism  # Import the function you want to test

class TestPlagiarismDetection(unittest.TestCase):
    """
    Probar que la funcion find_plagiarism detecta el plagio correctamente.
    """

    def setUp(self):
        """
        Inicializar textos y vectorizador para las pruebas.
        """

        self.entry_files = [('entry1.txt', 'This is a test entry text.'),
                            ('entry2.txt', 'Another test entry text.'),
                            ('entry3.txt', 'Another data text')
                           ]
        self.database_files = [('database1.txt', 'This is a database text.'),
                               ('database2.txt', 'Another database text.'),
                               ('database3.txt', 'Another test entry text.')
                             ]
        self.vectorizer = TfidfVectorizer()

        # Fit the vectorizer with database text
        all_texts = [text for _, text in self.database_files + self.entry_files]
        self.vectorizer.fit(all_texts)
        self.threshold = 0.7

    def test_no_plagiarism(self):
        """
        Caso de prueba para cuando no se encuentra plagio.
        """
        self.threshold = 0.7
        result = find_plagiarism([self.entry_files[0]], self.database_files, self.vectorizer, self.threshold)
        self.assertEqual(len(result), 0)

    def test_plagiarism(self):
        """
        Caso de prueba para cuando se encuentra plagio
        """
        self.threshold = 0.7
        result = find_plagiarism([self.entry_files[1]], self.database_files, self.vectorizer, self.threshold)
        self.assertEqual(len(result), 1) 

    def test_some_plagiarism(self):
        """
        Caso de prueba para cuando solo un porcentaje del texto es plagiado.
        """
        self.threshold = 0.3
        result = find_plagiarism([self.entry_files[2]], self.database_files, self.vectorizer, self.threshold)
        self.assertEqual(len(result), 2)

    def test_empty_files(self):
        """
        Sí no se envian archivos de entrada, la función debe devolver una lista vacía.
        """
        self.threshold = 0.3
        result = find_plagiarism([], self.database_files, self.vectorizer, self.threshold)
        self.assertEqual(len(result), 0)

    def test_missing_list(self):
        """ Si el argumento de archivos de entrada no es una lista, debe lanzar un TypeError."""
        self.threshold = 0.3
       
        with self.assertRaises(TypeError):
            find_plagiarism(1, self.database_files[0], self.vectorizer, self.threshold)