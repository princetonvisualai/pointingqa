
# coding: utf-8

# ## Code to check accuracy of Workers
# 
# * I will be aiming for ~5 responses per point. Workers will be evaluated based on consensus with other evaluators. 
# * For each HIT. Decide on a consensus set based on the mode for each response. If multiple modes exist, consider both to be "correct"
# * Ensure that at least 3 workers have responded to a given task.
# * Grade quality for each HIT based on accuracy per the consensus set


# Parse Results Data
import collections
import json
import argparse
import os

#Grading Functions
def hash_responses(responses):
    for response in responses:
        output = response['output']
        if isinstance(output, dict):
            continue
        hashed_output = {}
        for out in output:
            hashed_output[out.keys()[0]] = out.values()[0]
        response['output'] = hashed_output

def hit_id_to_response(responses):
    hit_ids = collections.defaultdict(list)
    for i,r in enumerate(responses):
        hit_ids[r['hit_id']].append(r)
    return hit_ids

def get_consensus_set(rel_resps, hit_id=None):
    '''rel_resps is a list of relevant responses, each having identical keys.'''
    assert len(rel_resps) > 0, "0 relevent responses given"
    shared_hid = hit_id
    if isinstance(shared_hid, type(None)):
        shared_hid = rel_resps[0]['hit_id']
    responses = []
    for resp in rel_resps:
        this_hid = resp['hit_id']
        assert this_hid == shared_hid, "This set of responses not unified. {} != {}".format(this_hid, shared_hid)
        responses.append(resp['output'])
    elems = responses[0].keys()
    consensus_set = {}
    for el in elems:
        answers = collections.defaultdict(int)
        for resp in responses:
            answers[resp[el]['answer']]+=1
        maxval = 0
        maxresp = []
        for k,v in answers.items():
            if v > maxval:
                maxval = v
                maxresp = [k]
            elif v == maxval:
                maxresp.append(k)
        consensus_set[el] = (maxval, maxresp)        
    return consensus_set
def worker_id_to_hits(responses):
    id2output = collections.defaultdict(dict)
    for resp in responses:
        worker_id = resp['worker_id']
        hid = resp['hit_id']
        output = resp['output']
        assignment_id = resp['assignment_id']
        id2output[worker_id][hid] = (assignment_id, output)
    return id2output
def grade_hit(predict, target):
    N = float(len(target))
    correct = 0.0
    for el, (num, gt) in target.items():
        response = predict[el]['answer']
        if response in gt:
            correct+=1.0
    return (correct, N)

def main():
    #Script to get results
    fname = args.scorefile
    outfile = args.outfile
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
    with open(fname, 'r') as f:
        #acc_sec = f.readline()
        responses = []
        for l in f.readlines():
            responses.append(json.loads(l.strip()))

    hash_responses(responses) # account for shuffled order of images for a given HIT
    hid2resp = hit_id_to_response(responses) # Segment into similar HITS
    consensus_sets = {} # Select the "correct" responses for each HIT
    for hid, resps in hid2resp.items():
        consensus_sets[hid] = get_consensus_set(resps, hid)
    worker2hits = worker_id_to_hits(responses)
    worker_scores = collections.defaultdict(dict)
    for worker, hits in worker2hits.items():
        for hid,(assignment_id, answers) in hits.items():
            correct, N = grade_hit(answers, consensus_sets[hid])
            worker_scores[worker][hid] = (assignment_id, correct/N)

    # Show Scores
    print(worker_scores)
    with open(outfile, 'w') as f:
        json.dump(worker_scores, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--scorefile', type=str, default='../results/results.txt', help="Name the worker_scores" 
                        "JSON file to evaluate")
    parser.add_argument('-o', '--outfile', type=str, default='worker_scores.json', help="Name the outfile to write" 
                        "assignment ids to be approved")
    args = parser.parse_args()
    main()

