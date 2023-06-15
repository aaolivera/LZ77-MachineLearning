import numpy as np
import math
import contextlib
import os

os.environ['TF_DETERMINISTIC_OPS'] = '1'
from Arithmetic import ArithmeticEncoder, ArithmeticDecoder
from BitInputOutputStream import BitInputStream, BitOutputStream
from NeuralNetwork import build_model, get_symbol, process, reset_seed, train

## Parameters
mode = 'both' #@param ["compress", "decompress", "both", "preprocess_only"]
path_to_original_file = "Decompressed/file3.txt"
path_to_decompressed_file = "Decompressed/decompressed.txt"
path_to_compressed_file = "Compressed/compressed.dat" 

"""## Compress"""
if mode != 'decompress':
  input_path = path_to_original_file

  text = open(input_path, 'rb').read()
  vocab = sorted(set(text))
  vocab_size = len(vocab)
  # Creating a mapping from unique characters to indexes.
  char2idx = {u:i for i, u in enumerate(vocab)}
  # int_list will contain the characters of the file.
  int_list = []
  for idx, c in enumerate(text):
    int_list.append(char2idx[c])

  # Round up to a multiple of 8 to improve performance.
  vocab_size = math.ceil(vocab_size/8) * 8
  file_len = len(int_list)
  print ('Length of file: {} symbols'.format(file_len))
  print ('Vocabulary size: {}'.format(vocab_size))

if mode == 'compress' or mode == 'both':
  original_file = path_to_original_file
  path_to_file = path_to_compressed_file
  with open(path_to_file, "wb") as out, contextlib.closing(BitOutputStream(out)) as bitout:
    length = len(int_list)
    # Write the original file length to the compressed file header.
    out.write(length.to_bytes(5, byteorder='big', signed=False))
    # write 256 bits to the compressed file header to keep track of the vocabulary.
    for i in range(256):
      if i in char2idx:
        bitout.write(1)
      else:
        bitout.write(0)
    enc = ArithmeticEncoder(32, bitout)
    process(True, length, vocab_size, enc, int_list)
  print("Compressed size:", os.path.getsize(path_to_file))

"""## Decompress"""
if mode == 'decompress' or mode == 'both':
  output_path = path_to_decompressed_file
  with open(path_to_file, "rb") as inp, open(output_path, "wb") as out:
    # Read the original file size from the header.
    length = int.from_bytes(inp.read()[:5], byteorder='big')
    inp.seek(5)
    # Create a list to store the file characters.
    output = [0] * length
    bitin = BitInputStream(inp)
    # we get the vocabulary from the file header.
    vocab = []
    for i in range(256):
      if bitin.read():
        vocab.append(i)
    vocab_size = len(vocab)
    # Round up to a multiple of 8 to improve performance.
    vocab_size = math.ceil(vocab_size/8) * 8
    dec = ArithmeticDecoder(32, bitin)
    process(False, length, vocab_size, dec, output)

    # Convert indexes back to the original characters.
    idx2char = np.array(vocab)
    for i in range(length):
      out.write(bytes((idx2char[output[i]],)))
