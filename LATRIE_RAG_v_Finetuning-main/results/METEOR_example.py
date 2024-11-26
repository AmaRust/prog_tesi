import nltk
from nltk.translate.meteor_score import meteor_score

references = [["this", "is", "an", "apple"], ["that", "is", "an", "apple"]]
candidate = ["an", "apple", "on", "this", "tree"]

score = meteor_score(references, candidate)
print("METEOR score: ", score)

#---------------------------------------

references = [["this", "is", "an", "apple"], ["that", "is", "a", "red", "fruit"]]
candidate = ["a", "black", "color", "tree"]

score = meteor_score(references, candidate)
print("METEOR score: ", score)

#---------------------------------------

references = [["this", "is", "an", "apple"], ["that", "is", "a", "red", "fruit"]]
candidate = ["this", "is", "an", "apple"]

score = meteor_score(references, candidate)
print("METEOR score: ", score)