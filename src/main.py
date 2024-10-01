from src.morphological_evaluator import MorphologicalEvaluator

if __name__ == "__main__":    
    tokenizers = ["subword_tokenizers_31/wordpiece"]
    evaluator = MorphologicalEvaluator(tokenizers_paths=tokenizers)
    evaluator.full_eval()