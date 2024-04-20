import unittest
from sklearn.feature_extraction.text import TfidfVectorizer
from main import detect_plagiarism  # Import the function you want to test

class TestPlagiarismDetection(unittest.TestCase):
    """
    Probar que la funcion find_plagiarism detecta el plagio correctamente.
    """

    def setUp(self):
        """
        Inicializar textos y vectorizador para las pruebas.
        """

        #Inicializar textos para pruebas
        self.entry_files = [('entry1.txt', 'This is a test entry text.'),
                            ('entry2.txt', 'Another test entry text.'),
                            ('entry3.txt', 'Another data text')
                        ]
        self.database_files = [('database1.txt', 'This is a database text.'),
                               ('database2.txt', 'Another database text.'),
                               ('database3.txt', 'Another test entry text.')
                            ]   

        self.second_entry_files = [('entry1.txt', 'This is a test entry text.')
                                ]
        
        self.second_database_files = [('database1.txt', 'This is a test entry text.'),
                               ('database2.txt', 'This is a test entry text.'),
                               ('database3.txt', 'This is a test entry text.'),
                               ('database4.txt', 'This is a test entry text.')
                            ]
        
        self.third_entry_files = [('entry1.txt', 'This is a test entry text.')
                            ]
        self.third_database_files = [('database1.txt', 'This is a test entry text.')
                            ]
        
        # Ajustar vectorizadores con el texto de los archivos

        self.vectorizer = TfidfVectorizer()
        all_texts = [text for _, text in self.database_files + self.entry_files]
        self.vectorizer.fit(all_texts)
        self.threshold = 0.7

        self.second_vectorizer = TfidfVectorizer()
        all_second_texts = [text for _, text in self.second_database_files + self.second_entry_files]
        self.second_vectorizer.fit(all_second_texts)
        self.second_threshold = 0.5

        self.third_vectorizer = TfidfVectorizer()
        all_third_texts = [text for _, text in self.third_database_files + self.third_entry_files]
        self.third_vectorizer.fit(all_third_texts)
        self.third_threshold = 0.6


    def test_no_plagiarism(self):
        """
        Caso de prueba para cuando un archivo no se encuentra plagio.
        """
        self.threshold = 0.7
        result = detect_plagiarism([self.entry_files[0]], self.database_files, self.vectorizer, self.threshold)
        self.assertEqual(len(result), 0)

    def test_plagiarism(self):
        """
        Caso de prueba para cuando en un archivo se encuentra plagio
        """
        self.threshold = 0.7
        result = detect_plagiarism([self.entry_files[1]], self.database_files, self.vectorizer, self.threshold)
        self.assertEqual(len(result), 1) 

    def test_some_plagiarism(self):
        """
        Caso de prueba para cuando solo un porcentaje del texto de un archivo es plagiado.
        """
        self.threshold = 0.3
        result = detect_plagiarism([self.entry_files[2]], self.database_files, self.vectorizer, self.threshold)
        self.assertEqual(len(result), 2)

    def test_multiple_matches(self):
        """
        Caso de prueba para revisar que el programa detecte múltiples plagios para un archivo de entrada.
        """
        self.threshold = 0.3
        result = detect_plagiarism([self.second_entry_files[0]], self.second_database_files, self.second_vectorizer, self.second_threshold)
        self.assertEqual(len(result), 4)

    def test_exact_match(self):
        """
        Caso de prueba para revisar que para dos archivos iguales, el programa detecte que son idénticos
        """
        self.threshold = 0.3
        result = detect_plagiarism([self.third_entry_files[0]], self.second_database_files, self.third_vectorizer, self.third_threshold)
        margin = 0.05 
        self.assertAlmostEqual(result[0]['similarity'], 1.0, delta = margin)
  
    def test_file_correctness(self):
        """Esta caso de prueba revisa que los archivos que recibe la función detect_plagiarism 
        sean correctos, es decir, que sean listas de tuplas con dos strings cada una.
        """
        self.threshold = 0.3

        #Revisar que al enviar un archivo que no sea una lista, se levante un TypeError
        with self.assertRaises(TypeError):
            detect_plagiarism(1, self.database_files[0], self.vectorizer, self.threshold)
        
        #Revisar que al enviar listas vacías, se levante un ValueError
        with self.assertRaises(ValueError):
            detect_plagiarism([], [], self.vectorizer, self.threshold)
        
        #Revisar que al enviar tuplas que no sean de dos strings, se levante un ValueError
        with self.assertRaises(ValueError):
            detect_plagiarism([(45, 47)], [(47, 45)], self.vectorizer, self.threshold)
        
        #Revisar que al enviar tuplas que no sean de dos elementos, se levante un ValueError
        with self.assertRaises(ValueError):
            detect_plagiarism([("txt", "txt")], [("txt")], self.vectorizer, self.threshold)
    

    def test_vectorizer_type(self):
        """Este caso de prueba revisa que el vectorizador que recibe la función detect_plagiarism 
        sea correcto, es decir, que sea un objeto de tipo TfidfVectorizer.
        """
        self.threshold = 0.3

        #Revisar que al enviar un vectorizador que no sea de tipo TfidfVectorizer, se levante un TypeError
        with self.assertRaises(TypeError):
            detect_plagiarism([("txt", "txt")], [("txt", "txt")], "vectorizer", self.threshold)

  