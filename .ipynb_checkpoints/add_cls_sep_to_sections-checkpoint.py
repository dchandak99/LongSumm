import os
import re
from tqdm import tqdm

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

txt_files_dir_path = 'LongSumm-data/extractive_summaries/combined_sections_txt/'
save_path = 'LongSumm-data/extractive_summaries/combined_sections_txt_with_cls_sep/'

p_dirs = os.listdir(txt_files_dir_path)

if '.DS_Store' in p_dirs:
    p_dirs.remove('.DS_Store')

for p in tqdm(p_dirs):
    os.mkdir(save_path+p)
    section_txt_files = os.listdir(txt_files_dir_path+p)
    for s_file in section_txt_files:
        with open(txt_files_dir_path+p+'/'+s_file, 'r') as f:
            txt = f.readlines()
  
        txt = ' '.join(txt)
        txt = re.sub('et al.', 'et al', txt)
        txt = re.sub('\n', '', txt)
        lines = txt.split('.')
        
        valid_lines = []
        for i in range(len(lines)):
            if lines[i] == '':
                continue
            valid_lines.append(lines[i].strip())
            
        for i in range(len(valid_lines)):
            valid_lines[i] = valid_lines[i] + '.' + ' ' + tokenizer.cls_token + ' ' + tokenizer.sep_token
            
        txt = ' '.join(valid_lines)
        
        with open(save_path+p+'/'+s_file, 'w') as f:
            f.write(txt)




