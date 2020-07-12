import os 

if not os.path.exists('../../LongSumm-data/extractive_summaries/combined_sections_ext_summaries'):
    os.mkdir('../../LongSumm-data/extractive_summaries/combined_sections_ext_summaries')
    dirs = os.listdir('../../LongSumm-data/extractive_summaries/combined_sections_txt_with_cls_sep/')
    for dir in dirs:
        os.mkdir('../../LongSumm-data/extractive_summaries/combined_sections_ext_summaries/'+dir)


