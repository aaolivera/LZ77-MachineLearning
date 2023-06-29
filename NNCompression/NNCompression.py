from asyncio.windows_events import NULL
import numpy as np
import math
import contextlib
import os

os.environ['TF_DETERMINISTIC_OPS'] = '1'
from Arithmetic import ArithmeticEncoder, ArithmeticDecoder
from BitInputOutputStream import BitInputStream, BitOutputStream
from NeuralNetwork import build_model, get_symbol, process, reset_seed, train


path_to_original_file = "Decompressed/file.txt"

def Comprimir():
    input_path = input("Path del archivo a comprimir (default: " + path_to_original_file + "):")
    if input_path == "":
      input_path = path_to_original_file
    
    #Apertura del archivo a comprimir y cargar sarta a memoria
    text = open(input_path, 'rb').read()
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    # Generamos diccionario para convertir el vocabulario de entrada al vocabulario numerico de la red neuronal
    char2idx = {u:i for i, u in enumerate(vocab)}
    # Convertimos la sarta de entrada en una nueva sarta numerica (int_list).
    int_list = []
    for idx, c in enumerate(text):
      int_list.append(char2idx[c])
    
    vocab_size = math.ceil(vocab_size/8) * 8
    file_len = len(int_list)
    print ('Tamaño del archivo original: {}'.format(file_len))
    print ('Vocabulario de entrada: {}'.format([chr(key) for key, value in char2idx.items()]))
    
    path_to_file = input_path + '.comprimido'
    # Creamos archivo de salida
    with open(path_to_file, "wb") as out, contextlib.closing(BitOutputStream(out)) as bitout:
        length = len(int_list)
        # Escribimos en archivo de salida el largo original y el diccionario de índices.
        out.write(length.to_bytes(5, byteorder='big', signed=False))
        for i in range(256):
            if i in char2idx:
                bitout.write(1)
            else:
                bitout.write(0)
        enc = ArithmeticEncoder(32, bitout)
        # Iniciamos red neuronal y procesamos la sarta
        process(True, length, vocab_size, enc, int_list)
    print("Tamaño comprimido: ", os.path.getsize(path_to_file), " Porcentaje de compresión: ", round((file_len-os.path.getsize(path_to_file))* 100/file_len, 2))

def Descomprimir():
    path_to_file = input("Path del archivo a descomprimir (default: " + path_to_original_file + '.comprimido' + "):")
    if path_to_file == "":
        path_to_file = path_to_original_file + '.comprimido'

    with open(path_to_file, "rb") as inp, open(path_to_file.replace('.comprimido','.descomprimido'), "wb") as out:
        # Leemos el largo original y el diccionario que mapea el vocabulario de entrada con vocabulario numerico de la red neuronal.
        length = int.from_bytes(inp.read()[:5], byteorder='big')
        inp.seek(5)
        bitin = BitInputStream(inp)
        vocab = []
        for i in range(256):
            if bitin.read():
                vocab.append(i)
        vocab_size = math.ceil(len(vocab)/8) * 8
        dec = ArithmeticDecoder(32, bitin)
        # Creamos un array del tamanio del archivo original para almacenar la salida de la red neuronal
        output = [0] * length
        process(False, length, vocab_size, dec, output)
        
        # Convertimos la sarta retornada al vocabulario de entrada.
        idx2char = np.array(vocab)
        # Escribir archivo de salida
        for i in range(length):
            out.write(bytes((idx2char[output[i]],)))
        print("Descompresión terminada")

## Menu
while True:
    print("/////////////////////////////////////")
    print("////Iniciando compresor neuronal/////")
    print("/////////////////////////////////////")
    mode = input("Quiere comprimir o decomprimir? (c/d):")
    
    if mode == 'c':
        Comprimir()
    
    if mode == 'd':
        Descomprimir()