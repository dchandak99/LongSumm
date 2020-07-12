import os
import re
import json
import numpy as np
from tqdm import tqdm_notebook
from collections import Counter

base_path = 'LongSumm-data/extractive_summaries/'

path_to_jsons = base_path + 'papers-jsons/'

p_jsons = os.listdir(path_to_jsons)

p_unread = []

section_1 = ['abstract']
section_2 = ['introduction', 'problem formulation', 'overview', 'problem definition']
section_3 = ['related work', 'background', 'preliminaries', 'related works', 'previous work', 'baseline models']
section_4 = ['conclusion', 'conclusions', 'discussion', 'conclusion and future work', 'analysis', 'inference', 'discussion and conclusion', 'future work', 'theoretical analysis', 'concluding remarks']
section_5 = ['experiments', 'experimental setup', 'experiment', 'setup', 'training details', 'implementation', 'hyperparameters', ]
section_6 = ['model', 'approach', 'method', 'methods', 'methodology', 'models', 'our approach', 'proposed method', 'model architecture', 'algorithm']
section_7 = ['experimental results', 'results', 'evaluation', 'error analysis', 'main results', 'results and analysis', 'human evaluation', 'experimental evaluation', 'empirical results', 'experiments and results']
section_8 = ['data', 'datasets', 'dataset', 'evaluation metrics']
remove_sections = ['acknowledgements', 'acknowledgments', 'acknowledgement', 'acknowledgment', 'appendix', 'appendices', 'a appendix', 'notation']

section_names = []
for p in tqdm_notebook(p_jsons):
    with open(path_to_jsons+p) as json_file:
        try:
            p_data = json.load(json_file)
        except UnicodeDecodeError:
            p_unread.append(p)
            continue
    
    p_sections = {}
    p_sections['name_of_paper'] = p_data['name'][:-4]
    if p_data['metadata']['sections'] is not None:
        for s in p_data['metadata']['sections']:
            if s['heading'] is None:
                s['heading'] = 'abstract'
            s_name = re.sub(' +', ' ', re.sub('[^a-z\s]', '', s['heading'].lower())).lstrip()
            
            if s_name in remove_sections:
                continue
            else:
                section_names.append(s_name)
                if s_name in section_1:
                    p_sections['abstract'] = s['text']
                    
                elif s_name in section_2:
                    p_sections['introduction'] = s['text']
                    
                elif s_name in section_3:
                    p_sections['related_work'] = s['text']
                
                elif s_name in section_4:
                    p_sections['conclusion'] = s['text']
                    
                elif s_name in section_5:
                    p_sections['experiments'] = s['text']
                    
                elif s_name in section_6:
                    p_sections['model'] = s['text']
                    
                elif s_name in section_7:
                    p_sections['results'] = s['text']
                    
                elif s_name in section_8:
                    p_sections['data'] = s['text']
                    
                else:
                    if 'other' in p_sections.keys():
                        p_sections['other'] = ' '.join([p_sections['other'], s['text']])
                        p_sections['other_section_titles'].append(s_name)
                    
                    else:
                        p_sections['other'] = s['text']
                        p_sections['other_section_titles'] = []
                        p_sections['other_section_titles'].append(s_name)
            
    with open('LongSumm-data/extractive_summaries/combined_sections/'+p_sections['name_of_paper']+'.json', 'w') as file:
        json.dump(p_sections, file)


