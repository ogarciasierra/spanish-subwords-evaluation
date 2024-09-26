from src.morphological_evaluator import MorphologicalEvaluator

if __name__ == "__main__":    
    tokenizers = ["subword_tokenizers_31\bpe", "subword_tokenizers_31/wordpiece"]
    evaluator = MorphologicalEvaluator()
    evaluator.eval_transformers_task3()
