import os
import argparse

def custom_replace(sent):
	sent = sent.replace('\n', '')
	sent = sent.replace('al.', 'al')

	sent = sent.replace('. ', '. [CLS] [SEP] ')

	return sent

def add_cls_sep(sent_list):
	sents = []
	for i in range(len(sent_list)):
		sents.append(custom_replace(sent_list[i]).strip())

	return ' [CLS] [SEP] '.join(sents)

def load_doc(load_path):
	with open(load_path, 'r') as f:
		doc = f.readlines()

	return doc

def save_doc(doc, save_path):
	with open(save_path, 'w') as f:
		f.write(doc)

def extract_talksumm_summary(path):
	raw_summ = load_doc(path)

	sents = []
	for i in range(len(raw_summ)):
		sents.append(raw_summ[i].split('\t')[2].replace('\n', '').strip())

	doc = ' '.join(sents)

	save_doc(doc, 'txt_data/target/test_1.txt')

def extract_paper_content(path):
	doc_names = os.listdir(path)
	docs = []
	for d_n in doc_names:
		docs.append(add_cls_sep(load_doc(path + d_n)))

	doc = ' [CLS] [SEP] '.join(docs)

	save_doc(doc, 'txt_data/source/test_1.txt')

if __name__ == '__main__':
	#parser = argparse.ArgumentParser() #Add CLI args to modify the script does

	#extract_talksumm_summary('txt_data/talksumm_summaries/Abstractive Document Summarization with a Graph-Based Attentional Neural Model.txt')

	extract_paper_content('txt_data/combined_sections_txt/Abstractive Document Summarization with a Graph-Based Attentional Neural Model/')


