import pandas as pd
from typing import List, Dict, Union, Optional
from pathlib import Path
from api_models.api_models import EmotionModel
from direct_models.direct_models import EmotionModelDirect

class EmotionAnalyzer:
    """
    A class to analyze emotions in phrases using different models
    Model options:
        1: BERT tiny emotion intent
        2: BERT base uncased emotion
        3: Albert base v2 emotion
        4: Distilbert base uncased emotion
    """
    def __init__(
        self, 
        csv_path: Union[str, Path], 
        model_number: int = 2,  # Default is BERT base
        use_direct: bool = False
    ) -> None:
        """
        Initialize the emotion analyzer, default model is BERT base and use API
        
        Args:
            csv_path (Union[str, Path]): Path to the CSV file with phrases
            model_number (int): Number of the model to use (1-4)
                1: BERT tiny emotion intent
                2: BERT base uncased emotion
                3: Albert base v2 emotion
                4: Distilbert base uncased emotion
            use_direct (bool): Whether to use direct model or API
        
        Raises:
            ValueError: If model number is invalid (must be 1-4)
            FileNotFoundError: If CSV file doesn't exist
        """
        csv_path = "../" + csv_path
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        model_map = {
            1: "bert_tiny",      
            2: "bert_base",      
            3: "albert",         
            4: "distilbert"      
        }
        
        if model_number not in model_map:
            raise ValueError("Model number must be between 1 and 4")
            
        model_name = model_map[model_number]
        if use_direct:
            self.model = EmotionModelDirect.create(model_name)
        else:
            self.model = EmotionModel.create(model_name)
            
        self.results: Optional[List[Dict]] = None

    def read_csv(self) -> pd.DataFrame:
        """
        Read the CSV file and validate its contents
        
        Returns:
            pd.DataFrame: DataFrame containing the phrases
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV is empty or malformed
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
        df = pd.read_csv(self.csv_path)
        
        if df.empty:
            raise ValueError("CSV file is empty")
            
        if 'phrase' not in df.columns:
            # Try to use first column if no 'phrase' column
            df = pd.read_csv(self.csv_path, header=None, names=['phrase'])
        return df

    def analyze_phrases(self) -> List[Dict]:
        """
        Read phrases from CSV file and get emotion predictions
        
        Returns:
            List[Dict]: List of dictionaries with phrases and predictions
        """
        df = self.read_csv()
        results = []

        total_phrases = len(df)
        for idx, row in df.iterrows():
            phrase = row['phrase']
            try:
                prediction = self.model.predict(phrase)
                results.append({
                    "phrase": phrase,
                    "prediction": prediction,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "phrase": phrase,
                    "prediction": None,
                    "status": f"error: {str(e)}"
                })
            
            # Print progress
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{total_phrases} phrases")

        self.results = results
        print(results)
        return results

    def save_results(self, output_path: Union[str, Path]) -> None:
        """
        Save analysis results to a CSV file
        
        Args:
            output_path (Union[str, Path]): Path where results will be saved
        """
        if self.results is None:
            self.results = self.analyze_phrases()
            
        output_path = Path(output_path)
        output_df = pd.DataFrame(self.results)
        output_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

    def print_results(self, limit: Optional[int] = None) -> None:
        """
        Print emotion analysis results to console
        
        Args:
            limit (Optional[int]): Maximum number of results to print
        """
        if self.results is None:
            self.results = self.analyze_phrases()
            
        results_to_print = self.results[:limit] if limit else self.results
        
        for result in results_to_print:
            print("\n" + "="*50)
            print(f"Phrase: {result['phrase']}")
            print(f"Status: {result['status']}")
            if result['prediction']:
                if isinstance(result['prediction'], dict):
                    print(f"Emotion: {result['prediction']['label']}")
                    print(f"Confidence: {result['prediction']['confidence']:.4f}")
                    print("\nAll probabilities:")
                    for emotion, prob in result['prediction']['all_predictions'].items():
                        print(f"{emotion}: {prob:.4f}")
                else:
                    print(f"Prediction: {result['prediction']}")

    def get_statistics(self) -> Dict[str, Union[int, float, Dict]]:
        """
        Calculate statistics about the analyzed phrases
        
        Returns:
            Dict[str, Union[int, float, Dict]]: Dictionary with statistics
        """
        if self.results is None:
            self.results = self.analyze_phrases()
            
        total = len(self.results)
        successful = sum(1 for r in self.results if r['status'] == 'success')
        failed = total - successful
        
        emotion_counts = {}
        confidence_sum = 0
        
        for result in self.results:
            if result['status'] == 'success':
                pred = result['prediction']
                if isinstance(pred, dict):
                    emotion = pred['label']
                    confidence = pred['confidence']
                else:
                    emotion = pred[0]['label']
                    confidence = pred[0]['score']
                    
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                confidence_sum += confidence
        
        return {
            "total_phrases": total,
            "successful_predictions": successful,
            "failed_predictions": failed,
            "success_rate": successful/total if total > 0 else 0,
            "average_confidence": confidence_sum/successful if successful > 0 else 0,
            "emotion_distribution": emotion_counts
        } 