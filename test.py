from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex

syn = []
def _refine(line):
    line = regex.sub("<[^>]+>", "", line)
    line = regex.sub("[^\s\p{Latin}']", "", line) 
    return line.strip()

sents = [regex.sub("[^a-zA-Z0-9\u4e00-\u9fa5\s]", "", line) for line in codecs.open(hp.test, 'r', 'utf-8').read().split("\n") if (line and line[0] != "<")]
a = sents[1].split()
a = a[-1]

for sent in sents:
    if sent[-1]==a:
        syn.append(sent)


o_sents = []

for each in syn:
    o_sents.append(each.split()[0])


for sent in o_sents[:10]:
	print(sent)
	for word in (sent).split():
		print(word)
		print(len(word))
		for each in word[:10]:
			print(each)
		print("</S>")
'''
syn = []
def _refine(line):
    line = regex.sub("<[^>]+>", "", line)
    line = regex.sub("[^a-zA-Z0-9\u4e00-\u9fa5\s']", "", line) 
    return line.strip()

sents = [_refine(line) for line in codecs.open(hp.train, 'r', 'utf-8').read().split("\n")]




'''
'''
a = sents[1].split()
a = a[-1]

for sent in sents:
	if sent[-1]==a:
		syn.append(sent)

for each in syn:
	print(each.split()[0])

'''
