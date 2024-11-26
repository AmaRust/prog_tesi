from rouge_score import rouge_scorer

reference_text = "The quick brown fox jumps over the lazy dog."
generated_text = "A fast brown fox jumps over a lazy dog."

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Compute ROUGE-L metric
scores = scorer.score(reference_text, generated_text)

print("ROUGE-L Score:", scores['rougeL'].fmeasure)

#---------------------------------------

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score('The quick brown fox jumps over the lazy dog',
                      'The quick brown dog jumps on the log.')

print(scores)