import json 
import codecs
from os.path import dirname, realpath
import sys

def load_json(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = json.load(file)
    return data

def dump_json(data, path):
    with open(path, 'w', encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)

def load_text(nombre_archivo):

    barra = '\\' if sys.platform == 'win32' else '/'
    directorio = dirname(realpath(__file__)) + barra
    path_archivo_texto = directorio + nombre_archivo
    with codecs.open(path_archivo_texto, mode='r', encoding='utf-8') as archivo_texto:
        texto_archivo = archivo_texto.read()
    return texto_archivo   


def calculate_metrics(gold, vocab):

        intersection = [element for element in gold if element in vocab]
        precision = len(intersection) / len(vocab)
        recall = len(intersection) / len(gold)
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        return {'precision': f"{precision * 100:.2f}%", 'recall': f"{recall* 100:.2f}%", 'f1': f"{f1 * 100:.2f}%"}
    
def post_process(word):
    d = {'Ã³': 'ó', 'ÃŃ': 'í', 'Ã¡':'á', 'Ãº':'ú', 'Ã©': 'é'}
    for key, value in d.items():
        word = word.replace(key, value)
    return word


def eliminar_repetidos(lista_de_listas):

    resultado = []
    conjunto_listas_vistas = set()

    for sublista in lista_de_listas:

        tupla_sublista = tuple(sublista)

        if tupla_sublista not in conjunto_listas_vistas:
            resultado.append(sublista)
            conjunto_listas_vistas.add(tupla_sublista)

    return resultado


def corregir_segmentacion(lista_segmentada):
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



