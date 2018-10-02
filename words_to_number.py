# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


data = _read_words('simple-examples/data/ptb.train.txt')
data = [' ' if '<' in word else word for word in data]

character_list = ' '.join(data)
character_list = list(character_list)
counter = collections.Counter(character_list)
count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
words, _ = list(zip(*count_pairs))
word_to_id = dict(zip(words, range(len(words))))
result = [word_to_id[word] for word in character_list if word in word_to_id]
sequence_size = 40
threshold = int(len(result)/sequence_size) * sequence_size
result = result[0:threshold]
import numpy as np
result = np.reshape(np.array(result), (-1, sequence_size))
n = np.shape(result)[0]
train_sentence = result[0:int(n*0.8),:]
test_sentence = result[int(n*0.8):,:]
np.savetxt('save/test.txt', test_sentence, fmt='%d')
np.savetxt('save/train.txt', train_sentence, fmt='%d')

translated = []
with open('save/eval_file.txt')as fin:
    for line in fin:
        line = line.strip()
        line = line.split()
        parse_line = ''.join([words[int(x)] for x in line])
        translated.append(parse_line)

