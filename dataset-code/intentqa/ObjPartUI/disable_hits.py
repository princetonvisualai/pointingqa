import argparse
import os
import sys
import simpleamt

if __name__ == '__main__':

  abs_path = os.path.dirname(os.path.abspath(__file__))
  print("FILE:\t{}".format(abs_path))
  parser = argparse.ArgumentParser(parents=[simpleamt.get_parent_parser()],
              description="Delete HITs")
  parser.add_argument('--all', action='store_true', default=False)
  args = parser.parse_args()

  if (args.hit_ids_file is not None) == args.all:
    print 'Must specify exactly one of --hit_ids_file or --all'
    sys.exit(1)

  mtc = simpleamt.get_mturk_connection_from_args(args)

  if args.all:
    hit_ids = []
    for hit in mtc.get_all_hits():
      hit_ids.append(hit.HITId)
  else:
    print("File path:\t", os.path.join(abs_path, args.hit_ids_file))
    with open(os.path.join(abs_path, args.hit_ids_file), 'r') as f:
      hit_ids = [line.strip() for line in f]

  print ('This will delete %d HITs with sandbox=%s'
         % (len(hit_ids), str(args.sandbox)))
  print 'Continue?'
  s = raw_input('(Y/N): ')
  if s == 'Y' or s == 'y':
    for index, hit_id in enumerate(hit_ids):
      try:
        mtc.disable_hit(hit_id)
        print 'disabling: %d / %d' % (index+1, len(hit_ids))
      except:
        print 'Failed to disable: %s' % (hit_id)
  else:
    print 'Aborting'
