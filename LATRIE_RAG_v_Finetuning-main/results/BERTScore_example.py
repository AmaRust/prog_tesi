import bert_score

# hide the loading messages
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

# Preparing the plot
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["xtick.major.size"] = 0
rcParams["xtick.minor.size"] = 0
rcParams["ytick.major.size"] = 0
rcParams["ytick.minor.size"] = 0

rcParams["axes.labelsize"] = "large"
rcParams["axes.axisbelow"] = True
rcParams["axes.grid"] = True

from bert_score import score

with open("../../test/data/hyps.txt") as f:
    cands = [line.strip() for line in f]

with open("../../test/data/refs.txt") as f:
    refs = [line.strip() for line in f]

print(cands[1])
print(refs[1])

# We obtain precision, recall and F1 scores for each sentence
P, R, F1 = score(cands, refs, lang='en', verbose=True)

print(P) # Precision
print(R) # Recall
print(F1)

# We take the average to obtain an overall score for each measure (P, R, F1)
print(f"System level F1 score: {F1.mean():.3f}")
print(f"System level Recall score: {R.mean():.3f}")
print(f"System level Precision score: {P.mean():.3f}")

plt.hist(F1, bins=20)
plt.xlabel("score")
plt.ylabel("counts")
plt.show()

# With "rescale_with_baseline=True" we can now see that the scores are much more spread out, which makes it easy to compare different examples.

P, R, F1 = score(cands, refs, lang='en', rescale_with_baseline=True)

plt.hist(F1, bins=20)
plt.xlabel("score")
plt.ylabel("counts")
plt.show()

# Another example with many references
single_cands = ['I like lemons.']
multi_refs = [['I am proud of you.', 'I love lemons.', 'Go go go.']]

P_mul, R_mul, F_mul = score(single_cands, multi_refs, lang="en", rescale_with_baseline=True)

print(F_mul) # the best score is chosen

# We can see similarity matrix, before and after rescaling
from bert_score import plot_example

plot_example(cands[0], refs[0], lang="en")

plot_example(cands[0], refs[0], lang="en", rescale_with_baseline=True)