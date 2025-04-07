import json
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
import os

class MetricsLogger:
    def __init__(self):
        self.metrics = {}
        
    def calculate_metrics(self, y_true, y_pred, dataset_name):
        """Calculate metrics for a set of predictions"""
        metrics = {}
        metrics['class_report'] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Convert numpy values to Python types for JSON serialization
        for cls in metrics['class_report']:
            for metric in metrics['class_report'][cls]:
                if isinstance(metrics['class_report'][cls][metric], np.float64):
                    metrics['class_report'][cls][metric] = float(metrics['class_report'][cls][metric])
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Calculate overall metrics
        metrics['overall_f1'] = float(f1_score(y_true, y_pred, average='weighted'))
        metrics['overall_accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # Store metrics
        self.metrics[dataset_name] = metrics
        
        return metrics
        
    def print_metrics(self, metrics, logger=None):
        """Print metrics in a readable format"""
        class_report = metrics['class_report']
        
        if logger:
            logger.info("\nClassification Report:")
            # Header
            logger.info(f"{'': <15}{'precision': <10}{'recall': <10}{'f1-score': <10}{'support': <10}")
            
            # Class metrics
            for cls in sorted([c for c in class_report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]):
                p = class_report[cls]['precision']
                r = class_report[cls]['recall']
                f1 = class_report[cls]['f1-score']
                s = class_report[cls]['support']
                logger.info(f"{cls: <15}{p: <10.2f}{r: <10.2f}{f1: <10.2f}{s: <10}")
            
            # Average metrics
            for avg in ['macro avg', 'weighted avg']:
                if avg in class_report:
                    p = class_report[avg]['precision']
                    r = class_report[avg]['recall']
                    f1 = class_report[avg]['f1-score']
                    s = class_report[avg]['support']
                    logger.info(f"{avg: <15}{p: <10.2f}{r: <10.2f}{f1: <10.2f}{s: <10}")
            
            # Confusion Matrix
            logger.info("\nConfusion Matrix:")
            cm = metrics['confusion_matrix']
            for row in cm:
                logger.info(' '.join(f'{x: <8}' for x in row))
            
            # Overall metrics
            logger.info(f"\nOverall F1 Score: {metrics['overall_f1']:.4f}")
            logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        else:
            print("\nClassification Report:")
            # Header
            print(f"{'': <15}{'precision': <10}{'recall': <10}{'f1-score': <10}{'support': <10}")
            
            # Class metrics
            for cls in sorted([c for c in class_report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]):
                p = class_report[cls]['precision']
                r = class_report[cls]['recall']
                f1 = class_report[cls]['f1-score']
                s = class_report[cls]['support']
                print(f"{cls: <15}{p: <10.2f}{r: <10.2f}{f1: <10.2f}{s: <10}")
            
            # Average metrics
            for avg in ['macro avg', 'weighted avg']:
                if avg in class_report:
                    p = class_report[avg]['precision']
                    r = class_report[avg]['recall']
                    f1 = class_report[avg]['f1-score']
                    s = class_report[avg]['support']
                    print(f"{avg: <15}{p: <10.2f}{r: <10.2f}{f1: <10.2f}{s: <10}")
            
            # Confusion Matrix
            print("\nConfusion Matrix:")
            cm = metrics['confusion_matrix']
            for row in cm:
                print(' '.join(f'{x: <8}' for x in row))
            
            # Overall metrics
            print(f"\nOverall F1 Score: {metrics['overall_f1']:.4f}")
            print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    
    def save_metrics(self, filename):
        """Save metrics to a JSON file"""
        # Get the models directory path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Ensure filename is just the basename (no path)
        filename = os.path.basename(filename)
        
        # Combine with models directory path
        full_path = os.path.join(models_dir, filename)
        
        with open(full_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        return full_path 