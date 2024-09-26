
from transformers import AutoTokenizer

class old():

    def guess_algorithm(self):

        task_1_data = self.task1  
        words = task_1_data['prefixes']


        bpe_tokenizer = AutoTokenizer.from_pretrained(self.bpe30_path)
        bpe_vocab = bpe_tokenizer.get_vocab()
        bpe_vocab = list(bpe_vocab.keys())
        bpe_vocab = [word for word in bpe_vocab if word in words]

        beto_tokenizer = AutoTokenizer.from_pretrained(self.beto_path)
        beto_vocab = beto_tokenizer.get_vocab()
        beto_vocab = list(beto_vocab.keys())
        beto_vocab = [word for word in beto_vocab if word in words]

        wp_tokenizer = AutoTokenizer.from_pretrained(self.wordpiece30_path)
        wp_vocab = wp_tokenizer.get_vocab()
        wp_vocab = list(wp_vocab.keys())
        wp_vocab = [word for word in wp_vocab if word in words]

        print('metric prefixes beto-bpe', self.calculate_metrics(beto_vocab, bpe_vocab))
        print('metric prefixes beto-wordpiece', self.calculate_metrics(beto_vocab, wp_vocab))


    def eval_rules_task2(self):


        all_words = self.task2

        path = 'rules'

        task_2_results = {}

        task_2_results['prefixes'] = {}
        task_2_results['stems'] = {}
        task_2_results['suffixes'] = {}
        task_2_results['clitics'] = {}
        

        print(f"Evaluating {path}")
        
        tokenizer = self.rules_tokenizer

        for i, morpheme_type in enumerate(all_words):

            print(f"Evaluating {morpheme_type}")
            words = all_words[morpheme_type]
            words = self.eliminar_repetidos(words)
            print(f"Total words {morpheme_type}:", len(words))

            monotoken = []
            multitoken_incorrect = []
            multitoken_correct = []

            morph_dict = {}
            morph_dict[morpheme_type] = {}

            for tuple in words:
                
                word = tuple[0]
                morph = tuple[1]

                candidates = tokenizer.tokenize(word)
                
                longest_pos = max(candidates, key=lambda clave: len(candidates[clave]))
                tokens = candidates[longest_pos]
                tokens = [t.text if t.text in self.vocab_38k else ['UNK'] for t in tokens]    

                if len(tokens) == 1:
                    monotoken.append([word, tokens])
                else:
                    if morph in tokens:
                        multitoken_correct.append([word, tokens])
                    else: 
                        multitoken_incorrect.append([word, tokens])

            task_2_results[f"{morpheme_type}"][f"{path}_total"] =   len(words)
            task_2_results[f"{morpheme_type}"][f"{path}_monotoken"] =    f"{(len(monotoken)/len(words)) * 100:.2f}%"
            task_2_results[f"{morpheme_type}"][f"{path}_multitoken correct"] =   f"{(len(multitoken_correct)/len(words)) * 100:.2f}%"
            task_2_results[f"{morpheme_type}"][f"{path}_multitoken incorrect"] =   f"{(len(multitoken_incorrect)/len(words)) * 100:.2f}%"
            
    
        print(task_2_results)

    
    def eval_joint_task2(self):


        all_words = self.task2

        path = 'joint'

        task_2_results = {}

        task_2_results['prefixes'] = {}
        task_2_results['stems'] = {}
        task_2_results['suffixes'] = {}
        task_2_results['clitics'] = {}
        

        print(f"Evaluating {path}")
        
        tokenizer = self.joint_tokenizer

        for i, morpheme_type in enumerate(all_words):

            print(f"Evaluating {morpheme_type}")
            words = all_words[morpheme_type]
            words = self.eliminar_repetidos(words)
            print(f"Total words {morpheme_type}:", len(words))

            monotoken = []
            multitoken_incorrect = []
            multitoken_correct = []

            morph_dict = {}
            morph_dict[morpheme_type] = {}

            for tuple in words:
                
                word = tuple[0]
                morph = tuple[1]

                tokens = tokenizer.tokenize(word)
                
                tokens = [t if t in self.vocab_38k else ['UNK'] for t in tokens]    

                if len(tokens) == 1:
                    monotoken.append([word, tokens])
                else:
                    if morph in tokens:
                        multitoken_correct.append([word, tokens])
                    else: 
                        multitoken_incorrect.append([word, tokens])

            task_2_results[f"{morpheme_type}"][f"{path}_total"] =   len(words)
            task_2_results[f"{morpheme_type}"][f"{path}_monotoken"] =    f"{(len(monotoken)/len(words)) * 100:.2f}%"
            task_2_results[f"{morpheme_type}"][f"{path}_multitoken correct"] =   f"{(len(multitoken_correct)/len(words)) * 100:.2f}%"
            task_2_results[f"{morpheme_type}"][f"{path}_multitoken incorrect"] =   f"{(len(multitoken_incorrect)/len(words)) * 100:.2f}%"
            
    
        print(task_2_results)

    def eval_rules_task1(self):

        task_1_data = self.task1  

        task_1_results = {}

        vocab = self.vocab_50k

        for morph_type in task_1_data:
                
            print(f"Evaluating {morph_type}")
            words = task_1_data[morph_type]
            print(f"Total datos {morph_type}:", len(words))
            if type(words[0]) == type(words):
                words = [word[0] for word in words]
                           
        
            metrics = self.calculate_metrics(words, vocab)
            task_1_results[morph_type] = metrics
        
        print(task_1_results)


    def eval_rules_task2(self):


        all_words = self.task2

        path = 'rules'

        task_2_results = {}

        task_2_results['prefixes'] = {}
        task_2_results['stems'] = {}
        task_2_results['suffixes'] = {}
        task_2_results['clitics'] = {}
        

        print(f"Evaluating {path}")
        
        tokenizer = self.rules_tokenizer

        for i, morpheme_type in enumerate(all_words):

            print(f"Evaluating {morpheme_type}")
            words = all_words[morpheme_type]
            words = self.eliminar_repetidos(words)
            print(f"Total words {morpheme_type}:", len(words))

            monotoken = []
            multitoken_incorrect = []
            multitoken_correct = []

            morph_dict = {}
            morph_dict[morpheme_type] = {}

            for tuple in words:
                
                word = tuple[0]
                morph = tuple[1]

                candidates = tokenizer.tokenize(word)
                
                longest_pos = max(candidates, key=lambda clave: len(candidates[clave]))
                tokens = candidates[longest_pos]
                tokens = [t.text if t.text in self.vocab_38k else ['UNK'] for t in tokens]    

                if len(tokens) == 1:
                    monotoken.append([word, tokens])
                else:
                    if morph in tokens:
                        multitoken_correct.append([word, tokens])
                    else: 
                        multitoken_incorrect.append([word, tokens])

            task_2_results[f"{morpheme_type}"][f"{path}_total"] =   len(words)
            task_2_results[f"{morpheme_type}"][f"{path}_monotoken"] =    f"{(len(monotoken)/len(words)) * 100:.2f}%"
            task_2_results[f"{morpheme_type}"][f"{path}_multitoken correct"] =   f"{(len(multitoken_correct)/len(words)) * 100:.2f}%"
            task_2_results[f"{morpheme_type}"][f"{path}_multitoken incorrect"] =   f"{(len(multitoken_incorrect)/len(words)) * 100:.2f}%"
            
    
        print(task_2_results)

    
    def eval_joint_task2(self):


        all_words = self.task2

        path = 'joint'

        task_2_results = {}

        task_2_results['prefixes'] = {}
        task_2_results['stems'] = {}
        task_2_results['suffixes'] = {}
        task_2_results['clitics'] = {}
        

        print(f"Evaluating {path}")
        
        tokenizer = self.joint_tokenizer

        for i, morpheme_type in enumerate(all_words):

            print(f"Evaluating {morpheme_type}")
            words = all_words[morpheme_type]
            words = self.eliminar_repetidos(words)
            print(f"Total words {morpheme_type}:", len(words))

            monotoken = []
            multitoken_incorrect = []
            multitoken_correct = []

            morph_dict = {}
            morph_dict[morpheme_type] = {}

            for tuple in words:
                
                word = tuple[0]
                morph = tuple[1]

                tokens = tokenizer.tokenize(word)
                
                tokens = [t if t in self.vocab_38k else ['UNK'] for t in tokens]    

                if len(tokens) == 1:
                    monotoken.append([word, tokens])
                else:
                    if morph in tokens:
                        multitoken_correct.append([word, tokens])
                    else: 
                        multitoken_incorrect.append([word, tokens])

            task_2_results[f"{morpheme_type}"][f"{path}_total"] =   len(words)
            task_2_results[f"{morpheme_type}"][f"{path}_monotoken"] =    f"{(len(monotoken)/len(words)) * 100:.2f}%"
            task_2_results[f"{morpheme_type}"][f"{path}_multitoken correct"] =   f"{(len(multitoken_correct)/len(words)) * 100:.2f}%"
            task_2_results[f"{morpheme_type}"][f"{path}_multitoken incorrect"] =   f"{(len(multitoken_incorrect)/len(words)) * 100:.2f}%"
            
    
        print(task_2_results)


    def eval_rules_task3(self):

        # task 3

        total = 0
        wrong = []
        correct = []
        
        for word in self.task3:

            total = total+1
            gold_segmentations = word[1]
            word = word[0]
            all_segmentations = []
            all_predictions = []

            for pos_tag in gold_segmentations:
                segmentation = gold_segmentations[pos_tag]
                all_segmentations.append(segmentation)

            predictions = self.rules_tokenizer.tokenize(word, overstem=False)

            for prediction in predictions:
                prediction = predictions[prediction]
                try:
                    prediction = [t.text for t in prediction] 
                    prediction = ['##'+t if i >0 else t for i,t in enumerate(prediction)]
                    #prediction = [t if t in self.vocab_50k else '[UNK]' for t in prediction] 
                    
                    all_predictions.append(prediction)
                except:
                    print(word, prediction)
            
            intersection = [list(sublista) for sublista in map(tuple, all_predictions) if tuple(sublista) in map(tuple, all_segmentations)]
            

            if len(intersection) > 0:
                correct.append((word, predictions))
                
            else:
                
                wrong.append((word, all_segmentations, all_predictions))
                print((word, all_segmentations, all_predictions))
              

        print(len(correct)/total)

        dump_json(wrong, 'rules_errors.json')


    def eval_joint_task3(self):

        # task 3

        total = 0
        wrong = []
        correct = []
        m = 0


        print(len(self.task3))
        for word in self.task3:
            
            total = total+1
            gold_segmentations = word[1]
            word = word[0]
            all_segmentations = []
            all_predictions = []

            for pos_tag in gold_segmentations:
                segmentation = gold_segmentations[pos_tag]
                all_segmentations.append(segmentation)

            all_predictions = self.joint_tokenizer.tokenize(word)
            all_predictions = [all_predictions]
            #all_predictions = [[t if t in self.vocab_38k else '[UNK]' for t in all_predictions]]
                
            
            intersection = [list(sublista) for sublista in map(tuple, all_predictions) if tuple(sublista) in map(tuple, all_segmentations)]

            if len(intersection) > 0:
                correct.append((word, all_predictions))
                if len(all_predictions[0]) == 1:
                    m+=1
            else:
                wrong.append((word, all_segmentations, all_predictions))
                print(word, all_segmentations, all_predictions)

        
        print(m)
        print(len(correct))

        print('Errores', len(wrong))
        
        print(len(correct)/total)
        
        
        dump_json(wrong, 'joint_errors.json')
    
    
    def error_analysis(self):

        # task 3

        words = [
                "deshojar", "libros", "población"
            ]


        
        for word in words:
            for path in self.transformer_models:
                tokenizer = AutoTokenizer.from_pretrained(path)
                vocab = tokenizer.get_vocab()
                vocab = list(vocab.keys())
                print(path, 'érrimo' in vocab, '##érrimo' in vocab)
                print(path+':', word, '-->', [self.post_process(word) for word in tokenizer.tokenize(word)])
                print()
        


    
    
    
    def check_task_3_vocab(self):    

        from collections import Counter

        tags = []

        new = []

        for word in load_json('data/evaluation/morph_segmentation.json'):

            out = {}
            for pos in word[1]:
                tags.append(pos)
                tokens = word[1][pos]
                tokens = ["##"+element if i>0 and not element.startswith('#') else element for i, element in enumerate(tokens)]
                out[pos] = tokens
            
            new.append([word[0], out])
        
        print(new)
        print(len(new))
        print(Counter(tags))

    

    def full_eval_task3(self):
        self.eval_transformers_task3()
        self.eval_rules_task3()
        self.eval_joint_task3()





import time
import re
from collections import Counter
import pandas as pd

import tqdm
from transformers import AutoTokenizer

from src.utils import load_json, dump_json

from src.tokenizer.rules_tokenizer import RulesTokenizer
from src.tokenizer.joint_tokenizer import JointTokenizer

class MorphologicalEvaluator:

    """
    task 1: morphological coherence of tokenizers in 5 prefixes, 5 stems, and 5 suffixes
    task 2: morphological quality of tokenizers: intersection among vocabs: all spanish prefixes, all spanish suffixes and 1000 spanish stems (or all verb stems)
    task 2: morphological segmentation of words from a gold_standard
    """

    def __init__(self):

        self.task1 = load_json('data\evaluation\morph_quality.json')
        self.task2 = load_json('data\evaluation\morph_coherence.json')
        self.task3 = load_json('data\evaluation\morph_segmentation.json')

        self.wordpiece30_path = "../subword_tokenizers_31/wordpiece"
        self.bpe30_path = "../subword_tokenizers_31/bpe"
        self.unigram30_path = "../subword_tokenizers_31/unigram"
        self.wordpiece50_path = "../subword_tokenizers_52/wordpiece"
        self.bpe50_path = "../subword_tokenizers_52/bpe"
        self.unigram50_path = "../subword_tokenizers_52/unigram"
        self.beto_path = "dccuchile/bert-base-spanish-wwm-uncased"
        self.roberta_path = "PlanTL-GOB-ES/roberta-base-bne"
        self.transformer_models = [self.wordpiece30_path, 
                                   self.beto_path,
                                   self.bpe30_path]
        
        
        
        

        self.rules_tokenizer = RulesTokenizer()
        self.joint_tokenizer = JointTokenizer()

        self.afixes = pd.read_csv('data/vocabs/afixes.csv', usecols=['FORMA', 'TIPO', 'DERIVADOS'])
        self.prefixes = self.afixes[self.afixes['TIPO'].str.startswith('pref')]['FORMA'].values.tolist()
        self.suffixes = self.afixes[self.afixes['TIPO'].str.startswith('suf')]['FORMA'].values.tolist()

        self.vocab_50k = load_json('52kvocab.json')
        self.vocab_38k = load_json('38kvocab.json')
        
        

    def calculate_metrics(self, gold, vocab):

        intersection = [element for element in gold if element in vocab]
        precision = len(intersection) / len(vocab)
        recall = len(intersection) / len(gold)
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0


        return {'precision': f"{precision * 100:.2f}%", 'recall': f"{recall* 100:.2f}%", 'f1': f"{f1 * 100:.2f}%"}
    
    def post_process(self, word):
        d = {'Ã³': 'ó', 'ÃŃ': 'í', 'Ã¡':'á', 'Ãº':'ú', 'Ã©': 'é'}
        for key, value in d.items():
            word = word.replace(key, value)
        return word
    
    def eliminar_repetidos(self, lista_de_listas):

        resultado = []
        conjunto_listas_vistas = set()

        for sublista in lista_de_listas:
            # Convertir la sublista a una tupla para verificar su unicidad
            tupla_sublista = tuple(sublista)
            
            # Verificar si la tupla no está en el conjunto de tuplas vistas
            if tupla_sublista not in conjunto_listas_vistas:
                resultado.append(sublista)
                conjunto_listas_vistas.add(tupla_sublista)

        return resultado
    

    def corregir_segmentacion(self, lista_segmentada):
        nueva_lista = []
        i = 0
        while i < len(lista_segmentada):
            token = lista_segmentada[i]
            if token == 'Ìģ' and i > 0:
                token_anterior = nueva_lista.pop()
                if token_anterior[-1] in ['a', 'e', 'i', 'o', 'u']:
                    token_anterior = token_anterior[:-1] + 'áéíóú'[('aeiou').index(token_anterior[-1])]
                nueva_lista.append(token_anterior + lista_segmentada[i + 1])
                i += 1
            else:
                nueva_lista.append(token)
            i += 1

        return nueva_lista

    

    def eval_transformers_task1(self):

        """task 1: morphological quality of tokenizers: intersection among vocabs: all spanish prefixes, all spanish suffixes and 1000 spanish stems (or all verb stems)"""

        ## corregir ##, etc probar en temp_builder
        ## comparar sufijis: csv y reglas

        task_1_data = self.task1  

        task_1_results = {}

        bpe = []
        beto = []
        wordpiece = []

        for path in tqdm.tqdm(self.transformer_models):  
            print(f"Evaluating {path}")
            results = {}
            tokenizer = AutoTokenizer.from_pretrained(path)
            vocab = tokenizer.get_vocab()
            vocab = list(vocab.keys())

                      
            vocab = [self.post_process(word) for word in vocab]
            
            if 'bpe' in path or 'bne' in path:
                    vocab = ["##"+element if not element.startswith("Ġ") and not element.startswith("#") else element[1:] for i, element in enumerate(vocab)]

            elif 'unigram' in path:
                                
                vocab = ["##"+element if not element.startswith("▁") and not element.startswith("#") else element[1:] for i, element in enumerate(vocab)]

            all = []
            for morph_type in task_1_data:
                
                print(f"Evaluating {morph_type}")
                words = task_1_data[morph_type]
                all = all+words
                print(f"Total datos {morph_type}:", len(words))
                if type(words[0]) == type(words):
                    all = all + [word[0] for word in words]
                
            
            
            metrics = self.calculate_metrics(all, vocab)
            results[morph_type] = metrics

            task_1_results[path] = results
        
        dump_json(task_1_results, "data/results/task_1_tot_results.json")

    def guess_algorithm(self):

        bpe_tokenizer = AutoTokenizer.from_pretrained(self.bpe30_path)
        bpe_vocab = bpe_tokenizer.get_vocab()
        bpe_vocab = list(bpe_vocab.keys())

        beto_tokenizer = AutoTokenizer.from_pretrained(self.beto_path)
        beto_vocab = beto_tokenizer.get_vocab()
        beto_vocab = list(beto_vocab.keys())

        wp_tokenizer = AutoTokenizer.from_pretrained(self.wordpiece30_path)
        wp_vocab = wp_tokenizer.get_vocab()
        wp_vocab = list(wp_vocab.keys())

        print(self.calculate_metrics(beto_vocab,bpe_vocab))
        print(self.calculate_metrics(beto_vocab,wp_vocab ))



    def eval_rules_task1(self):

        task_1_data = self.task1  

        task_1_results = {}

        vocab = self.vocab_50k

        for morph_type in task_1_data:

            words = task_1_data[morph_type]
            if type(words[0]) == type(words):
                
                words = [word[0] for word in words]
                           
        
            metrics = self.calculate_metrics(words, vocab)
            task_1_results[morph_type] = metrics
        
        print(task_1_results)


    def eval_transformers_task2(self):

        ## corregir ##, etc probar en temp_builder
        ## comparar sufijis: csv y reglas

        """task 2: morphological coherence of tokenizers in prefixes, stems, suffixes and clitics"""

        all_words = self.task2

        task_2_results = {}

        task_2_results['prefixes'] = {}
        task_2_results['stems'] = {}
        task_2_results['suffixes'] = {}
        task_2_results['clitics'] = {}
        
        for path in tqdm.tqdm(self.transformer_models):
            print(f"Evaluating {path}")
            tokenizer = AutoTokenizer.from_pretrained(path)

            results_dict = {}

            monotoken = []
            multitoken_incorrect = []
            multitoken_correct = []

            all = []

            for i, morpheme_type in enumerate(all_words):

                
                words = all_words[morpheme_type]
                words = self.eliminar_repetidos(words)
                all = all+words

            print(len(all))
            for tuple in all:
                
                word = tuple[0]
                morph = tuple[1]
                
                try:
                    tokens = tokenizer.tokenize(word)
                except:
                    tokens = tokenizer.tokenize(tuple[0][0])

                if 'unigram' in path:
                    tokens = ["##"+element if i>0 else element[1:] for i, element in enumerate(tokens)]
                elif 'bpe' in path or 'bne' in path:
                    tokens = ["##"+element if i>0 else element for i, element in enumerate(tokens)]
                    tokens = self.corregir_segmentacion(tokens)

                tokens = [self.post_process(word) for word in tokens]

                if len(tokens) == 1:
                    monotoken.append([word, tokens])
                else:
                    if morph in tokens:
                        multitoken_correct.append([word, tokens])
                    else: 
                        multitoken_incorrect.append([word, tokens])

            task_2_results[f"{path}_total"] =   len(all)
            task_2_results[f"{path}_monotoken"] =    f"{(len(monotoken)/len(all)) * 100:.2f}%"
            task_2_results[f"{path}_multitoken correct"] =   f"{(len(multitoken_correct)/len(all)) * 100:.2f}%"
            task_2_results[f"{path}_multitoken incorrect"] =   f"{(len(multitoken_incorrect)/len(all)) * 100:.2f}%"
                
                
        dump_json(task_2_results, 'data/results/task_2_results_totals.json')


    def eval_rules_task2(self):


        all_words = self.task2

        path = 'rules'

        task_2_results = {}

        task_2_results['prefixes'] = {}
        task_2_results['stems'] = {}
        task_2_results['suffixes'] = {}
        task_2_results['clitics'] = {}
        

        print(f"Evaluating {path}")
        
        tokenizer = self.rules_tokenizer

        for i, morpheme_type in enumerate(all_words):

            print(f"Evaluating {morpheme_type}")
            words = all_words[morpheme_type]
            words = self.eliminar_repetidos(words)
            print(f"Total words {morpheme_type}:", len(words))

            monotoken = []
            multitoken_incorrect = []
            multitoken_correct = []

            morph_dict = {}
            morph_dict[morpheme_type] = {}

            for tuple in words:
                
                word = tuple[0]
                morph = tuple[1]

                candidates = tokenizer.tokenize(word)
                
                longest_pos = max(candidates, key=lambda clave: len(candidates[clave]))
                tokens = candidates[longest_pos]
                tokens = [t.text if t.text in self.vocab_38k else ['UNK'] for t in tokens]    

                if len(tokens) == 1:
                    monotoken.append([word, tokens])
                else:
                    if morph in tokens:
                        multitoken_correct.append([word, tokens])
                    else: 
                        multitoken_incorrect.append([word, tokens])

            task_2_results[f"{morpheme_type}"][f"{path}_total"] =   len(words)
            task_2_results[f"{morpheme_type}"][f"{path}_monotoken"] =    f"{(len(monotoken)/len(words)) * 100:.2f}%"
            task_2_results[f"{morpheme_type}"][f"{path}_multitoken correct"] =   f"{(len(multitoken_correct)/len(words)) * 100:.2f}%"
            task_2_results[f"{morpheme_type}"][f"{path}_multitoken incorrect"] =   f"{(len(multitoken_incorrect)/len(words)) * 100:.2f}%"
            
    
        print(task_2_results)

    
    def eval_joint_task2(self):


        all_words = self.task2

        path = 'joint'

        task_2_results = {}

        task_2_results['prefixes'] = {}
        task_2_results['stems'] = {}
        task_2_results['suffixes'] = {}
        task_2_results['clitics'] = {}
        

        print(f"Evaluating {path}")
        
        tokenizer = self.joint_tokenizer

        for i, morpheme_type in enumerate(all_words):

            print(f"Evaluating {morpheme_type}")
            words = all_words[morpheme_type]
            words = self.eliminar_repetidos(words)
            print(f"Total words {morpheme_type}:", len(words))

            monotoken = []
            multitoken_incorrect = []
            multitoken_correct = []

            morph_dict = {}
            morph_dict[morpheme_type] = {}

            for tuple in words:
                
                word = tuple[0]
                morph = tuple[1]

                tokens = tokenizer.tokenize(word)
                
                tokens = [t if t in self.vocab_38k else ['UNK'] for t in tokens]    

                if len(tokens) == 1:
                    monotoken.append([word, tokens])
                else:
                    if morph in tokens:
                        multitoken_correct.append([word, tokens])
                    else: 
                        multitoken_incorrect.append([word, tokens])

            task_2_results[f"{morpheme_type}"][f"{path}_total"] =   len(words)
            task_2_results[f"{morpheme_type}"][f"{path}_monotoken"] =    f"{(len(monotoken)/len(words)) * 100:.2f}%"
            task_2_results[f"{morpheme_type}"][f"{path}_multitoken correct"] =   f"{(len(multitoken_correct)/len(words)) * 100:.2f}%"
            task_2_results[f"{morpheme_type}"][f"{path}_multitoken incorrect"] =   f"{(len(multitoken_incorrect)/len(words)) * 100:.2f}%"
            
    
        print(task_2_results)





    def eval_tokenizer(self, path):

        # task 3

        total = 0
        wrong = []
        correct = []
        m = 0

        l = 0

        tokenizer = AutoTokenizer.from_pretrained(path)
        
        for word in self.task3:
            
            total = total+1
            gold_segmentations = word[1]
            word = word[0]
            all_segmentations = []
            all_predictions = []

            for pos_tag in gold_segmentations:
                segmentation = gold_segmentations[pos_tag]
                all_segmentations.append(segmentation)

            all_predictions = tokenizer.tokenize(word)

            l = l+ len(all_predictions)


            all_predictions = [re.sub('##','', pred) for pred in all_predictions]
            all_predictions = [self.post_process(word) for word in all_predictions]
            all_predictions = [[re.sub('▁','', pred) for pred in all_predictions]]

            
            
            intersection = [list(sublista) for sublista in map(tuple, all_predictions) if tuple(sublista) in map(tuple, all_segmentations)]

            if len(intersection) > 0:
                correct.append((word, all_predictions))
                if len(all_predictions[0]) == 1:
                    m+=1
            else:
                wrong.append((word, all_predictions))
        print(m)
        print(m/(len(correct)))
        print(len(correct))
        
        print('media:',l/1231)

        return len(correct)/total, wrong


    def eval_transformers_task3(self):

        # task 3

        results = {}
        for path in self.transformer_models:
            print(f"Evaluating {path}")
            score, wrongs = self.eval_tokenizer(path)
            results[path] = score
        
        print(results)

    def eval_rules_task3(self):

        # task 3

        total = 0
        wrong = []
        correct = []
        
        for word in self.task3:

            total = total+1
            gold_segmentations = word[1]
            word = word[0]
            all_segmentations = []
            all_predictions = []

            for pos_tag in gold_segmentations:
                segmentation = gold_segmentations[pos_tag]
                all_segmentations.append(segmentation)

            predictions = self.rules_tokenizer.tokenize(word, overstem=False)

            for prediction in predictions:
                prediction = predictions[prediction]
                try:
                    prediction = [t.text for t in prediction] 
                    prediction = ['##'+t if i >0 else t for i,t in enumerate(prediction)]
                    #prediction = [t if t in self.vocab_50k else '[UNK]' for t in prediction] 
                    
                    all_predictions.append(prediction)
                except:
                    print(word, prediction)
            
            intersection = [list(sublista) for sublista in map(tuple, all_predictions) if tuple(sublista) in map(tuple, all_segmentations)]
            

            if len(intersection) > 0:
                correct.append((word, predictions))
                
            else:
                
                wrong.append((word, all_segmentations, all_predictions))
                print((word, all_segmentations, all_predictions))
              

        print(len(correct)/total)

        dump_json(wrong, 'rules_errors.json')


    def eval_joint_task3(self):

        # task 3

        total = 0
        wrong = []
        correct = []
        m = 0


        print(len(self.task3))
        for word in self.task3:
            
            total = total+1
            gold_segmentations = word[1]
            word = word[0]
            all_segmentations = []
            all_predictions = []

            for pos_tag in gold_segmentations:
                segmentation = gold_segmentations[pos_tag]
                all_segmentations.append(segmentation)

            all_predictions = self.joint_tokenizer.tokenize(word)
            all_predictions = [all_predictions]
            #all_predictions = [[t if t in self.vocab_38k else '[UNK]' for t in all_predictions]]
                
            
            intersection = [list(sublista) for sublista in map(tuple, all_predictions) if tuple(sublista) in map(tuple, all_segmentations)]

            if len(intersection) > 0:
                correct.append((word, all_predictions))
                if len(all_predictions[0]) == 1:
                    m+=1
            else:
                wrong.append((word, all_segmentations, all_predictions))
                print(word, all_segmentations, all_predictions)

        
        print(m)
        print(len(correct))

        print('Errores', len(wrong))
        
        print(len(correct)/total)
        
        
        dump_json(wrong, 'joint_errors.json')
    
    
    def error_analysis(self):

        # task 3

        words = [
                "rehaces"
            ]


        
        for word in words:
            for path in self.transformer_models:
                tokenizer = AutoTokenizer.from_pretrained(path)
                vocab = tokenizer.get_vocab()
                vocab = list(vocab.keys())
                print(path, 're' in vocab, 'hac' in vocab, 'es' in vocab)
                print(path+':', word, '-->', [self.post_process(word) for word in tokenizer.tokenize(word)])
                print()
        


    
    
    
    def check_task_3_vocab(self):    

        from collections import Counter

        tags = []

        new = []

        for word in load_json('data/evaluation/morph_segmentation.json'):

            out = {}
            for pos in word[1]:
                tags.append(pos)
                tokens = word[1][pos]
                tokens = ["##"+element if i>0 and not element.startswith('#') else element for i, element in enumerate(tokens)]
                out[pos] = tokens
            
            new.append([word[0], out])
        
        print(new)
        print(len(new))
        print(Counter(tags))

    

    def full_eval_task3(self):
        self.eval_transformers_task3()
        self.eval_rules_task3()
        self.eval_joint_task3()


if __name__ == "__main__":    
    start = time.time()
    evaluator = MorphologicalEvaluator()
    evaluator.guess_algorithm()
    end = time.time()
    res = end - start
    print('Execution time:', res, 'seconds')
