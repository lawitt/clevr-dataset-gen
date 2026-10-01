"""
Run this once, after every parallel render_images.py job has finished, to
produce two combined outputs:

  1. A single --output_scene_file combining every per-scene JSON file in
     --output_scene_dir (same logic render_images.py itself does at the end
     of a single, non-parallel run).
  2. A single combined camera_info.jsonl file merging every per-job
     camera_info_<start>_<end>.jsonl file, deduplicated by (scene_index,
     view_index) just in case two jobs' ranges ever overlapped.

This is a plain Python script -- no Blender required. Run it with the
system/venv Python, e.g.:

    python merge_outputs.py \\
        --output_scene_dir /path/to/output/scenes \\
        --output_scene_file /path/to/output/CLEVR_scenes.json \\
        --camera_info_dir /path/to/output \\
        --combined_camera_info_file /path/to/output/camera_info.jsonl
"""
import argparse
import glob
import json
import os
from datetime import datetime as dt

parser = argparse.ArgumentParser()
parser.add_argument('--output_scene_dir', required=True,
    help="Same --output_scene_dir all the render_images.py jobs wrote to.")
parser.add_argument('--output_scene_file', required=True,
    help="Where to write the combined single JSON scene file.")
parser.add_argument('--filename_prefix', default='CLEVR')
parser.add_argument('--split', default='new')
parser.add_argument('--version', default='1.0')
parser.add_argument('--license', default="Creative Commons Attribution (CC-BY 4.0)")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"))
parser.add_argument('--camera_info_dir', required=True,
    help="Directory containing the per-job camera_info_<start>_<end>.jsonl files.")
parser.add_argument('--combined_camera_info_file', required=True,
    help="Where to write the combined camera_info.jsonl file.")


def merge_scene_jsons(args):
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
  all_scene_paths = sorted(glob.glob(os.path.join(args.output_scene_dir, prefix + '*.json')))
  all_scenes = []
  for scene_path in all_scene_paths:
    with open(scene_path, 'r') as f:
      all_scenes.append(json.load(f))
  output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': args.split,
      'license': args.license,
    },
    'scenes': all_scenes,
  }
  with open(args.output_scene_file, 'w') as f:
    json.dump(output, f)
  print("Combined %d scene files into %s" % (len(all_scenes), args.output_scene_file))


def merge_camera_info(args):
  camera_info_paths = sorted(glob.glob(os.path.join(args.camera_info_dir, 'camera_info_*.jsonl')))
  seen = set()
  n_written = 0
  with open(args.combined_camera_info_file, 'w') as out_f:
    for path in camera_info_paths:
      with open(path, 'r') as in_f:
        for line in in_f:
          line = line.strip()
          if not line:
            continue
          try:
            entry = json.loads(line)
          except ValueError:
            continue
          key = (entry['scene_index'], entry['view_index'])
          if key in seen:
            continue
          seen.add(key)
          out_f.write(json.dumps(entry) + '\n')
          n_written += 1
  print("Merged %d per-job camera-info files (%d entries total) into %s" % (
    len(camera_info_paths), n_written, args.combined_camera_info_file))


if __name__ == '__main__':
  args = parser.parse_args()
  merge_scene_jsons(args)
  merge_camera_info(args)