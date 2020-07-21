"""
Automatic evaluation functions
"""

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score


def translate(sentence):
  result, attention_weights = evaluate(sentence)
  
  return tokenizer_en.decode([i for i in result if i < tokenizer_en.vocab_size])  

def get_preds(r, t):
  generated = []
  real = []
  for r_, t_ in zip(r,t):
    generated.append(translate(r_))
    real.append(t_)
  return generated, real



def _get_ngrams(n, text):
  """Calculates n-grams.
  Args:
    n: which n-grams to calculate
    text: An array of tokens
  Returns:
    A set of n-grams
  """
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram_set.add(tuple(text[i:i + n]))
  return ngram_set


def rouge_n(eval_sentences, ref_sentences, n=2):
  """Computes ROUGE-N f1 score of two text collections of sentences.
  Source: https://www.microsoft.com/en-us/research/publication/
  rouge-a-package-for-automatic-evaluation-of-summaries/
  Args:
    eval_sentences: The sentences that have been picked by the summarizer
    ref_sentences: The sentences from the reference set
    n: Size of ngram.  Defaults to 2.
  Returns:
    f1 score for ROUGE-N
  """

  f1_scores = []
  for eval_sentence, ref_sentence in zip(eval_sentences, ref_sentences):
    eval_ngrams = _get_ngrams(n, eval_sentence)
    ref_ngrams = _get_ngrams(n, ref_sentence)
    ref_count = len(ref_ngrams)
    eval_count = len(eval_ngrams)

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = eval_ngrams.intersection(ref_ngrams)
    overlapping_count = len(overlapping_ngrams)

    # Handle edge case. This isn't mathematically correct, but it's good enough
    if eval_count == 0:
      precision = 0.0
    else:
      precision = overlapping_count / eval_count

    if ref_count == 0:
      recall = 0.0
    else:
      recall = overlapping_count / ref_count

    f1_scores.append(2.0 * ((precision * recall) / (precision + recall + 1e-8)))

  # return overlapping_count / reference_count
  return np.mean(f1_scores, dtype=np.float32)




def bleu_score(real, g):
  score = []
  for ref, cand in zip(real, g):
    cand = cand.split()
    ref = [ref.split()]
    #print(ref, cand)
    
    score.append(sentence_bleu(ref, cand))

  return sum(score)/len(score)


def meteor_score(real_, g_):
  score = []
  for ref, cand in zip(real_, g_):

    score.append(single_meteor_score(ref, cand))

  return sum(score)/len(score)


def eval_metrics(real, gen):
  rogue = rouge_n(gen, real)
  blue = bleu_score(real, gen)
  meteor = meteor_score(real, gen)
  print('rogue:',rogue, '\nBleu:',blue,'\nMeteor:', meteor)
  return rogue, blue, meteor

