import pandas as pd
from typing import List, Dict, Union, Optional, Literal
from pathlib import Path
from models.emotion_models import EmotionModel

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
        model_name: Literal["BERT_TINY", "BERT_BASE", "ALBERT", "DISTILBERT"] = "BERT_BASE", 
        method: Literal["pipeline", "direct"] = "pipeline",
        use_gpu: bool = True
    ) -> None:
        """
        Initialize the emotion analyzer, default model is BERT base and use API
        
        Args:
            csv_path (Union[str, Path]): Path to the CSV file with phrases
            model_name (str, optional): Name of the model to use. Defaults to "BERT_BASE".
                options: "BERT_TINY", "BERT_BASE", "ALBERT", "DISTILBERT"
            method (str, optional): Method to use for predictions. Defaults to "pipeline".
                options: "pipeline", "direct"
            use_gpu (bool, optional): Whether to use GPU if available. Defaults to True.
                If True but no GPU is available, will fallback to CPU.
        
        Raises:
            FileNotFoundError: If CSV file doesn't exist
        """
        
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        self.csv_path = csv_path

        self.model = EmotionModel(model_name, method, use_gpu)
        
        self.results: Optional[List[Dict]] = None

    #TODO: REVIEW
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
            
        # Get the name of the first column
        first_column_name = df.columns[0]
        
        return df, first_column_name
    
    #TODO: REVIEW AND TAKE CARE OF LIMITS
    def analyze_phrases(self) -> List[Dict]:
        """
        Read phrases from CSV file and get emotion predictions
        
        Returns:
            List[Dict]: List of dictionaries with phrases and predictions
        """
        df, column_name = self.read_csv()
        results = []

        total_phrases = len(df)
        for idx, row in df.iterrows():
            phrase = row[column_name]
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
        return results

    def print_results(self, limit: Optional[int] = None):
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
            if result['prediction'] :
                print(f"Label: {result['prediction']['label']}")
                print(f"Score: {result['prediction']['score']:.4f}")

    #FIXME
    def save_results(self, output_path: Union[str, Path]):
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

    #FIXME
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