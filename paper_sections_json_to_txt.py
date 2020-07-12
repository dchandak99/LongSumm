import os
import json
from tqdm import tqdm

base_path = 'LongSumm-data/extractive_summaries/'
path_to_jsons = base_path + 'combined_sections/'
txt_files_dir_path = base_path + 'combined_sections_txt/'

jsons = os.listdir(path_to_jsons)

p_skipped = []

for p in tqdm(jsons):
    with open(path_to_jsons+p) as json_file:
        p_data = json.load(json_file)
    os.mkdir(txt_files_dir_path+p_data['name_of_paper'])
    for key in p_data.keys():
        if key == 'other_section_titles' or key == 'name_of_paper':
            continue
        else:
            with open(txt_files_dir_path+p_data['name_of_paper']+'/'+key+'.txt', 'w') as f:
                try:
                    f.write(p_data[key])
                except(UnicodeEncodeError):
                    p_skipped.append(p)

"python train.py -task ext -mode test_text -text_src ../raw_data/txt_files/text2.txt -result_path ../results/output_2.txt -test_from ../models/bertext_cnndm_transformer.pt visible_gpus -1"
