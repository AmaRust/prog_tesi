import nltk
from nltk.translate.bleu_score import sentence_bleu

hypothesis = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']
reference = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']

print(sentence_bleu([reference], hypothesis, ))

#---------------------------------------

reference1 = [
    'this is a dog'.split(),
    'it is dog'.split(),
    'dog it is'.split(),
    'a dog, it is'.split() 
]

reference2 = [
    'this is a dog'.split()
]

candidate = 'it is a dog'.split()

print('Individual 1-gram: %f' % sentence_bleu(reference2, candidate, weights=(1, 0, 0, 0)))
print('Individual 2-gram: %f' % sentence_bleu(reference2, candidate, weights=(0, 1, 0, 0)))
print('Individual 3-gram: %f' % sentence_bleu(reference2, candidate, weights=(0, 0, 1, 0)))
print('Individual 4-gram: %f' % sentence_bleu(reference2, candidate, weights=(0, 0, 0, 1)))