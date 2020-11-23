# coding: utf-8
import json
import argparse
import os

def main(args):
    fname = args.scoreFile
    thresh = args.threshold
    outfile = args.outfile
    rejfile = args.rejectfile
    try:
        abs_path = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        abs_path = os.path.dirname(os.path.abspath('.'))
    except: #give up
        print("Abs Path Incorrect")
        abs_path = os.getcwd() 
    if fname[0] != '/':
        fname = os.path.join(abs_path, fname)
    if outfile[0] != '/':
        outfile = os.path.join(abs_path, outfile)
    if rejfile[0] != '/':
        rejfile = os.path.join(abs_path, rejfile)
    print("Getting scores from {}".format(fname))
    with open(fname, 'r') as f:
        data = json.load(f)
    approved = []
    rejected = []
    print("Evaluating Scores")
    for worker_id, hits in data.items():
        for hid, (asnid, score) in hits.items():
            if score >= thresh:
                approved.append(asnid)
            else:
                rejected.append(asnid)

    print("Scores evaluated. Outputting results to {}".format(outfile))
    with open(outfile, 'w') as f:
        for asnid in approved:
            f.write("{}\n".format(asnid))

    print("Outputting rejected assignments to {}".format(rejfile))
    with open(rejfile, 'w') as f:
        for asnid in rejected:
            f.write("{}\n".format(asnid))
    
                
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threshold', type=int, default=0.85, help="Select the accuracy threshold"
                        "for assignments you wish to approve")
    parser.add_argument('-f', '--scoreFile', type=str, default='worker_scores.json', help="Name the worker_scores" 
                        "JSON file to evaluate")
    parser.add_argument('-o', '--outfile', type=str, default='assignment_ids.txt', help="Name the outfile to write" 
                        "assignment ids to be approved")
    parser.add_argument('-r', '--rejectfile', type=str, default='rejected_assignment_ids.txt', help="Name the outfile to write"
                        "assignment ids to be rejected")
    args = parser.parse_args()
    main(args)

