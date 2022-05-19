import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import pandas as pd
from scripts.modeling import Modeler


class TestCases(unittest.TestCase):
    
    df = pd.read_csv("data/data.csv")
    analyzer = Modeler(df)
    numeric_pipeline = analyzer.generate_pipeline("numeric")
    numeric_transformation =  analyzer.generate_transformation(numeric_pipeline,"numeric","number")
    numerical_features = analyzer.store_features("numeric","number")
    
    def test_generate_pipeline(self):
        """
        Test that Mlscript generates a pipeline
        """
        self.assertTrue(self.numeric_pipeline)
    
    
    def test_store_features(self):
        """
        - testing store features
        """
        
        self.assertTrue(self.numerical_features)
