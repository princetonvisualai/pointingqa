#!/bin/bash
ids_file=examples/weak/hit_ids.txt
#Clean Rendered Templates
for f in $(ls rendered_template*.html);
do 
    rm $f
done

#Disable Current HIT
echo 'y' | python disable_hits.py --hit_ids_file=${ids_file}

# Remove the ids file
cmd="rm ${ids_file}" 
echo ${cmd}
${cmd}
#input_cache=input/understanding_pointing/im_names_short.json
#input_data=input/understanding_pointing/understanding_pointing_input_short.json
#input_cache=input/understanding_pointing/im_names_batched.json
#input_data=input/understanding_pointing/understanding_pointing_input_batched.json
#input_cache=input/understanding_pointing/clswise_im_names_batched.json
#input_data=input/understanding_pointing/clswise_data_batched.json # ^^ Discontinued ^^
#input_data=input/understanding_pointing/clswise_balanced_data.json # valid but randomly choose point within all parts
#input_data=input/understanding_pointing/clswise_balanced_data_medial.json # uses the medial axis skeleton for silhouettes
input_data=input/understanding_pointing/clswise_balanced_data_medial_centered_all.json # uses the centered medial axis skeleton for all classes
#input_data=input/understanding_pointing/BYCLASS/BYNAME_horse_centered.json
#input_data=input/understanding_pointing/BYCLASS/BYNAME_person_centered.json
#input_data=input/understanding_pointing/BYCLASS/BYNAME_bicycle_centered.json     
#input_data=input/understanding_pointing/BYCLASS/BYNAME_motorbike_centered.json
#input_data=input/understanding_pointing/BYCLASS/BYNAME_aeroplane_centered.json   
#input_data=input/understanding_pointing/BYCLASS/BYNAME_bottle_centered.json
#input_data=input/understanding_pointing/BYCLASS/BYNAME_bus_centered.json         
#input_data=input/understanding_pointing/BYCLASS/BYNAME_bird_centered.json        
#input_data=input/understanding_pointing/BYCLASS/BYNAME_pottedplant_centered.json
#input_data=input/understanding_pointing/BYCLASS/BYNAME_car_centered.json 
#input_data=input/understanding_pointing/BYCLASS/BYNAME_sheep_centered.json
#input_data=input/understanding_pointing/BYCLASS/BYNAME_dog_centered.json
input_data=input/understanding_pointing/BYCLASS/BYNAME_cat_centered.json
#input_data=input/understanding_pointing/BYCLASS/BYNAME_train_centered.json
#input_data=input/understanding_pointing/BYCLASS/BYNAME_cow_centered.json

# With input cache
cmd1="python launch_hits.py \
  --html_template=all_actions.html \
  --hit_properties_file=hit_properties/weak.json \
  --input_json_file=${input_data} \
  --input_cache=${input_cache} \
  --hit_ids_file=examples/weak/hit_ids.txt"
#Test case
cmd2="python launch_hits.py \
  --html_template=all_actions.html \
  --hit_properties_file=hit_properties/weak.json \
  --input_json_file=examples/image_sentence/example_input.txt \
  --hit_ids_file=examples/weak/hit_ids.txt"
#No input cache now
cmd3="python launch_hits.py \
  --html_template=all_actions.html \
  --hit_properties_file=hit_properties/weak.json \
  --input_json_file=${input_data} \
  --hit_ids_file=examples/weak/hit_ids.txt"
cmd=${cmd3}
echo ${cmd} 
${cmd}

