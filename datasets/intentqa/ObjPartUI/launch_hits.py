from __future__ import print_function
import os
import sys
import argparse
import json
import inspect
import simpleamt
from boto.mturk.price import Price
from boto.mturk.question import HTMLQuestion
from boto.mturk.connection import MTurkRequestError


def printPlus(*args):
    print(inspect.getouterframes(inspect.currentframe())[1][2], ": ", args)


DEBUG = printPlus
# Motorbike 20-40
# Dog 40-70
# Person 40-70

# Real Deal
MINHITS = 0
MAXHITS = 20

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[simpleamt.get_parent_parser()])
    parser.add_argument('--hit_properties_file', type=argparse.FileType('r'))
    parser.add_argument('--html_template')
    parser.add_argument('--input_json_file', type=argparse.FileType('r'))
    parser.add_argument('--input_cache', type=argparse.FileType('r'))
    args = parser.parse_args()

    im_names = []
    if args.input_cache is not None:
        # DEBUG("Cache: {}".format(args.input_cache))
        for i, line in enumerate(args.input_cache):
            im_names.append(json.loads(line.strip()))
        # im_names = json.load(args.input_cache)

    input_json_file = []
    for i, line in enumerate(args.input_json_file):
        input_json_file.append(line)

    mtc = simpleamt.get_mturk_connection_from_args(args)

    hit_properties = json.load(args.hit_properties_file)
    hit_properties['reward'] = Price(hit_properties['reward'])
    # hit_properties['Reward'] = str(hit_properties['Reward']).decode('utf-8')
    simpleamt.setup_qualifications(hit_properties, mtc)
    # DEBUG("After", hit_properties)

    frame_height = hit_properties.pop('frame_height')
    env = simpleamt.get_jinja_env(args.config)
    template = env.get_template(args.html_template)

    if args.hit_ids_file is None:
        DEBUG('Need to input a hit_ids_file')
        sys.exit()
    DEBUG(args.hit_ids_file, args.input_cache)
    if os.path.isfile(args.hit_ids_file):
        DEBUG('hit_ids_file already exists')
        sys.exit()

    with open(args.hit_ids_file, 'w') as hit_ids_file:
        # for i, line in enumerate(args.input_json_file):
        DEBUG("Launching {} HITS".format(len(input_json_file)))
        for i, line in enumerate(input_json_file):
            if i < MINHITS:
                continue
            hit_input = json.loads(line.strip())
            # In a previous version I removed all single quotes from the
            # json dump.
            # TODO: double check to see if this is still necessary.
            template_params = {'input': json.dumps(hit_input)}
            if len(im_names) > 0:
                template_params['im_names'] = json.dumps(
                    im_names[i])  # json.dumps(im_names)
            html = template.render(template_params)
            html_question = HTMLQuestion(html, frame_height)
            hit_properties['question'] = html_question
            # DEBUG('Rendering Template {}'.format(i))
            # with open('rendered_template{}.html'.format(i), 'w+') as f:
            #  f.write(html)
            # This error handling is kinda hacky.
            # TODO: Do something better here.
            if i >= MAXHITS:
                DEBUG(
                    "Debugging mode ON. Limiting HIT number to {}".format(
                        MAXHITS - MINHITS))
                break
            # continue # Don't actually launch
            launched = False
            while not launched:
                try:
                    boto_hit = mtc.create_hit(**hit_properties)
                    launched = True
                except MTurkRequestError as e:
                    DEBUG(e)
            hit_id = boto_hit[0].HITId
            hit_ids_file.write('%s\n' % hit_id)
            DEBUG('Launched HIT ID: %s, %d' % (hit_id, i + 1))
