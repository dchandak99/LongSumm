# stopwords = pkgutil.get_data(__package__, 'smart_common_words.txt')
# stopwords = stopwords.decode('ascii').split('\n')
# stopwords = {key.strip(): 1 for key in stopwords}
import os
import re
import shutil
import time

#from others import pyrouge

def test_rouge(temp_dir, cand, ref):
	candidates = [line.strip() for line in open(cand, encoding='utf-8')]
	references = [line.strip() for line in open(ref, encoding='utf-8')]
	print(len(candidates))
	print(len(references))
	assert len(candidates) == len(references)

	cnt = len(candidates)
	current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
	tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
	if not os.path.isdir(tmp_dir):
		os.mkdir(tmp_dir)
		os.mkdir(tmp_dir + "/candidate")
		os.mkdir(tmp_dir + "/reference")
	try:

		for i in range(cnt):
			if len(references[i]) < 1:
				continue
			with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
					  encoding="utf-8") as f:
				f.write(candidates[i])
			with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
					  encoding="utf-8") as f:
				f.write(references[i])
		r = pyrouge.Rouge155(temp_dir=temp_dir)
		r.model_dir = tmp_dir + "/reference/"
		r.system_dir = tmp_dir + "/candidate/"
		r.model_filename_pattern = 'ref.#ID#.txt'
		r.system_filename_pattern = r'cand.(\d+).txt'
		rouge_results = r.convert_and_evaluate()
		print(rouge_results)
		results_dict = r.output_to_dict(rouge_results)
	finally:
		pass
		if os.path.isdir(tmp_dir):
			shutil.rmtree(tmp_dir)
	return results_dict


def _get_ngrams(n, text):
	"""Calcualtes n-grams.

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


def _get_word_ngrams(n, sentences):
	"""Calculates word n-grams for multiple sentences.
	"""
	assert len(sentences) > 0
	assert n > 0

	# words = _split_into_words(sentences)

	words = sum(sentences, [])
	# words = [w for w in words if w not in stopwords]
	return _get_ngrams(n, words)