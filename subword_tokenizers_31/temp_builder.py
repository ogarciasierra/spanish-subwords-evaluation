from transformers import AutoTokenizer, AutoConfig

wordpiece_path = "subword_tokenizers/wordpiece"
bpe_path = "subword_tokenizers/bpe"
unigram_path = "subword_tokenizers/unigram"

transformer_models = [bpe_path]

d = {'Ã³': 'ó', 'ÃŃ': 'í', 'Ã¡':'á', 'Ãº':'ú', 'Ã©': 'é'}


def corregir_palabra(word, dict_corr):
    for key, value in dict_corr.items():
        word = word.replace(key, value)
    return word


text= "a todás nosotrés nos camión estín cuándo las gatas, las habitaciones y los reflujos"

for path in transformer_models:
    tokenizer = AutoTokenizer.from_pretrained(path)
    config = AutoConfig.from_pretrained(path)
    out = tokenizer.tokenize(text)


    # Corregir cada palabra segmentada usando el diccionario
    palabras_corregidas = [corregir_palabra(word, d) for word in out]

    # Resultado final
    print(palabras_corregidas)

