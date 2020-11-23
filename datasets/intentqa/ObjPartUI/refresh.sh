#!/bin/bash

cmd1=./examples/image_sentence/disable_hits.sh
echo ${cmd1}
${cmd1}
read -p 'Ready to remove? ' remove
echo $yes
rm ./examples/image_sentence/hit_ids.txt
./examples/image_sentence/launch_hits.sh
echo 'done'



