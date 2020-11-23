#!/bin/bash
echo ${BASH_SOURCE[0]}
echo $(dirname ${BASH_SOURCE[0]})
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo ${DIR}
echo "Evaluating Responses"
python ${DIR}/evaluate_responses.py
echo "Evaluating Assignments"
python ${DIR}/evaluate_assignments.py
