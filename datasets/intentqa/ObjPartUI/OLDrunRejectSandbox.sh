#!/bin/bash
echo "Rejecting assignments"
python reject_assignments.py --assignment_ids_file=examples/weak/evaluations/rejected_assignment_ids.txt
