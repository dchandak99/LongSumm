python bash_helpers/make_folders.py
for p_folder in ../../LongSumm-data/extractive_summaries/combined_sections_txt_with_cls_sep/*
do
	path_to_results_dir=$(echo "$p_folder" | cut -f1-4 -d"/")
	for file in "$p_folder"/*.txt
	do
		
		result_folder=$(echo "$file" | cut -f6 -d"/")
		result_path="$path_to_results_dir/combined_sections_ext_summaries/$result_folder/"
		section_name=$(echo "$file" | cut -f7 -d"/")
		#file=$(echo "$file" | sed 's/ /\\ /g')
		#result_path=$(echo "$result_path" | sed 's/ /\\ /g')
		result_path="${result_path}${section_name}"
		#echo $result_path

		python train.py -task ext -mode test_text -text_src "$file" -test_from ../models/bertext_cnndm_transformer.pt -visible_gpus -1 -result_path "$result_path"
			
	done
	break
done

