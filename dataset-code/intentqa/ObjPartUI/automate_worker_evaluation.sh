#!/bin/bash

echo "Evaluating Workers"
sh examples/weak/evaluations/automate_evaluation.sh
echo "Approving accepted assignments."
sh runApproveSandbox.sh
echo "Rejecting thresholded assignments"
sh runRejectSandbox.sh
