# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile, glob
from datetime import datetime as dt
from collections import Counter
import re
from pathlib import Path
import os

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  import mathutils
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  try:
    import utils
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.py.") 
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
SCRIPT_DIR = Path(__file__).resolve().parent

parser.add_argument('--base_scene_blendfile', default=SCRIPT_DIR / 'data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default=SCRIPT_DIR / 'data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default=SCRIPT_DIR / 'data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default=SCRIPT_DIR / 'data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument('--min_objects', default=3, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=10, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.25, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=400, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument('--max_retries', default=50, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of scenes to render during this job/session")
parser.add_argument('--num_viewpoints', default=25, type=int,
    help="The number of viewpoints per image to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--output_image_dir', default=SCRIPT_DIR / 'output/images',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default=SCRIPT_DIR / 'output/scenes',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default=SCRIPT_DIR / 'output/CLEVR_scenes.json',
    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default=SCRIPT_DIR / 'output/blendfiles',
    help="The directory where blender scene files will be stored, if the " +
         "user requested that these files be saved using the " +
         "--save_blendfiles flag; in this case it will be created if it does " +
         "not already exist.")
parser.add_argument('--camera_info_file', default=None,
    help="Path to a JSON-lines (.jsonl) file recording the azimuth and tilt " +
         "(elevation) angle of the camera used for each rendered image, one " +
         "JSON object per line. Run merge_outputs.py once after " +
         "all parallel jobs finish to combine the per-job files into one.")
parser.add_argument('--skip_final_merge', type=int, default=0,
    help="If set to 1, skip combining all per-scene JSON files in " +
         "--output_scene_dir into --output_scene_file at the end of this " +
         "run. Set this to 1 in every job when submitting multiple parallel, run " +
         "merge_outputs.py once after every parallel job has finished.")
parser.add_argument('--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=512, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=512, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=128, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")

def cleanup_unused_data():
    """Clean up unused Blender data to free memory"""
    import gc
    gc.collect()
    
    # Clear unused meshes
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    
    # Clear unused materials
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    
    # Clear unused textures
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    
    # Clear unused images
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    
    # Clear unused objects
    for block in bpy.data.objects:
        if block.users == 0:
            bpy.data.objects.remove(block)

def generate_scene_objects(args, scene_idx, num_objects, seed_add=0, _depth=0):
  if _depth > 10:
    if num_objects > args.min_objects:
      print("Warning: Could not place ", num_objects, " objects in scene ", scene_idx, ", retrying with different random seed")
      return generate_scene_objects(args, scene_idx, num_objects, seed_add= seed_add + 1, _depth=0)
    
  random.seed(scene_idx + seed_add)
  
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    size_mapping = list(properties['sizes'].items())
    
    shape_color_combos = None
    if args.shape_color_combos_json is not None:
      with open(args.shape_color_combos_json, 'r') as f:
        shape_color_combos = list(json.load(f).items())
        
    positions = []
    objects = []
    
    cardinal_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    attempts = 0
    while len(objects) < num_objects:
      attempts += 1
      if attempts > args.max_retries * num_objects:
        
        return generate_scene_objects.__wrapped__(args, scene_idx, num_objects, seed_add=args.num_viewpoints + seed_add +1, _depth=_depth+1)
      
      size_name, r = random.choice(size_mapping)
      x = random.uniform(-3, 3)
      y = random.uniform(-3, 3)
      
      dists_good = True
      margins_good = True
      for (xx, yy, rr) in positions:
        dx, dy = x - xx, y - yy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist - r - rr < args.min_dist:
          dists_good = False
          break
        for (dvx, dvy) in cardinal_dirs:
          margin = dx * dvx + dy * dvy
          if 0 < margin < args.margin:
            margins_good = False
            break
        if not margins_good:
          break
        
      if not (dists_good and margins_good):
        continue
      
      if shape_color_combos is None:
        obj_name, obj_name_out = random.choice(object_mapping)
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))
      else:
        obj_name_out, color_choices = random.choice(shape_color_combos)
        color_name = random.choice(color_choices)
        obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
        rgba = color_name_to_rgba[color_name]
        
      actual_r = r / math.sqrt(2) if obj_name == 'Cube' else r
      theta = 360.0 * random.random()
      mat_name, mat_name_out = random.choice(material_mapping)
      
      positions.append((x, y, actual_r))
      objects.append({
        'obj_name': obj_name,
        'obj_name_out': obj_name_out,
        'size_name': size_name,
        'r': r,
        'x': x,
        'y': y,
        'color_name': color_name,
        'rgba': rgba,
        'mat_name': mat_name,
        'mat_name_out': mat_name_out,
        'theta': theta,
      })
      
  return objects

  
generate_scene_objects.__wrapped__ = generate_scene_objects

def generate_viewpoint_angles(num_viewpoints, scene_idx):
  """
  generate random azimuth angles with a minimal gap between them,
  so not all are in a small area but well distributed and still random
  """
  even_spacing = (2 * math.pi) / num_viewpoints
  min_gap = even_spacing / 2.0
  
  random.seed(scene_idx + 99999)
  
  max_attempts = 1000
  for _ in range(max_attempts):
    angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(num_viewpoints)])
    
    gaps = [angles[i+1] - angles[i] for i in range(len(angles) - 1)]
    gaps.append((2 * math.pi - angles[-1]) + angles[0])
    
    if all(g >= min_gap for g in gaps):
      return angles
    
  start = random.uniform(0, even_spacing)
  return [start + even_spacing * i for i in range(num_viewpoints)]


def find_resume_point(args, prefix):
  """
  find the resume point for situations were a job maybe crashed or was interrupted,
  also useful when deciding later to generate an even bigger dataset,
  return last image and viewpoint to start generation from there
  """
  range_start = args.start_idx
  range_end = args.start_idx + args.num_images  # exclusive

  if not os.path.isdir(args.output_image_dir):
    return range_start, 0

  pattern = re.compile(r"^%s(\d+)_view_(\d+)\.png$" % re.escape(prefix))

  views_by_scene = {}
  for filename in os.listdir(args.output_image_dir):
    match = pattern.match(filename)
    if match:
      s_idx = int(match.group(1))
      if not (range_start <= s_idx < range_end):
        continue 
      v_idx = int(match.group(2))
      views_by_scene.setdefault(s_idx, set()).add(v_idx)

  if not views_by_scene:
    # Nothing rendered yet, start fresh
    return range_start, 0

  max_scene = max(views_by_scene.keys())
  views_done = views_by_scene[max_scene]

  if len(views_done) >= args.num_viewpoints:
    # move to the next one
    return max_scene + 1, 0

  # find the first missing view index
  next_view = 0
  while next_view in views_done:
    next_view += 1
  return max_scene, next_view


def load_camera_info_keys(path):
  """
  Reads an existing camera_info.jsonl file (if present) and returns the set
  of (scene_index, view_index) pairs already recorded.
  """
  keys = set()
  if not os.path.isfile(path):
    return keys
  with open(path, 'r') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        entry = json.loads(line)
      except ValueError:
        continue
      keys.add((entry['scene_index'], entry['view_index']))
  return keys


def append_camera_info(path, entry, camera_info_seen):
  """
  Appends a single camera-info record as one JSON line. Skips writing if this
  (scene_index, view_index) has already been recorded.
  """
  key = (entry['scene_index'], entry['view_index'])
  if key in camera_info_seen:
    return
  with open(path, 'a') as f:
    f.write(json.dumps(entry) + '\n')
    f.flush()
    os.fsync(f.fileno())
  camera_info_seen.add(key)


def main(args):
  num_digits = 6
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
  img_template = os.path.join(args.output_image_dir, img_template)
  scene_template = os.path.join(args.output_scene_dir, scene_template)
  blend_template = os.path.join(args.output_blend_dir, blend_template)

  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)

  # Resolve where the azimuth/tilt log lives,
  # different --start_idx/--num_images never write to the same file.
  camera_info_path = args.camera_info_file
  if camera_info_path is None:
    camera_info_path = os.path.join(
      os.path.dirname(os.path.normpath(args.output_scene_dir)),
      'camera_info_%06d_%06d.jsonl' % (args.start_idx, args.start_idx + args.num_images))
  camera_info_seen = load_camera_info_keys(camera_info_path)
  if camera_info_seen:
    print("Loaded %d existing camera-info entries from %s" % (
      len(camera_info_seen), camera_info_path))
  else:
    print("Will write new camera-info log to %s" % camera_info_path)

  # Figure out where to resume within THIS job's own range
  scene_idx, start_view = find_resume_point(args, prefix)
  range_end = args.start_idx + args.num_images  # exclusive upper bound of this job's scenes

  if scene_idx >= range_end:
    print("All %d scenes in this job's range [%d, %d) are already rendered; nothing to do." % (
      args.num_images, args.start_idx, range_end))
  else:
    if scene_idx != args.start_idx or start_view != 0:
      print("Resuming from scene %d, view %d" % (scene_idx, start_view))

    first_scene = True
    while scene_idx < range_end:
      random.seed(scene_idx)
      num_objects = random.randint(args.min_objects, args.max_objects)
      fixed_objects = generate_scene_objects(args, scene_idx, num_objects)
      angles = generate_viewpoint_angles(args.num_viewpoints, scene_idx)

      # only first scene can continue at any view number, upcoming scenes always start at view 0
      view_start = start_view if first_scene else 0
      first_scene = False

      for v in range(view_start, args.num_viewpoints):
        view_suffix = "_view_%d" % v
        img_path = (img_template % scene_idx).replace('.png', view_suffix + '.png')
        scene_path = (scene_template % scene_idx).replace('.json', view_suffix + '.json')
        blend_path = (blend_template % scene_idx).replace('.blend', view_suffix + '.blend') if args.save_blendfiles else None

        render_scene(args,
          num_objects=num_objects,
          output_index=scene_idx,
          output_split=args.split,
          output_image=img_path,
          output_scene=scene_path,
          output_blendfile=blend_path,
          view_idx=v,
          total_views=args.num_viewpoints,
          fixed_objects=fixed_objects,
          azimuth=angles[v],
          camera_info_path=camera_info_path,
          camera_info_seen=camera_info_seen,
        )

      cleanup_unused_data()
      scene_idx += 1

  if args.skip_final_merge:
    print("Skipping final scene-JSON merge (--skip_final_merge=1). "
          "Run merge_outputs.py once after all parallel jobs finish.")
  else:
    # Combine the JSON files for every scene rendered so far (not just this
    # job) into a single JSON file
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
      'scenes': all_scenes
    }
    with open(args.output_scene_file, 'w') as f:
      json.dump(output, f)
    
    
def look_at(obj, target):
  # calculates the rotation needed for the camera to face the center
  direction = target - obj.location
  rot_quat = direction.to_track_quat('-Z', 'Y')
  obj.rotation_euler = rot_quat.to_euler()


def render_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    output_blendfile=None,
    view_idx=0,
    total_views=1,
    fixed_objects=None,
    azimuth=0.0,
    camera_info_path=None,
    camera_info_seen=None,
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
  
  world = bpy.context.scene.world
  world.use_nodes = True
  nodes = world.node_tree.nodes
  links = world.node_tree.links
  
  ground = bpy.data.objects.get('Ground')
  if ground is not None:
    ground_material = ground.data.materials[0] if ground.data.materials else None
    bpy.context.scene.objects.unlink(ground)
    bpy.data.objects.remove(ground)
    
  bpy.ops.mesh.primitive_plane_add(radius=10000, location=(0, 0, 0))
  new_ground = bpy.context.object
  new_ground.name = 'Ground'
  
  white_mat = bpy.data.materials.new(name='Ground_White')
  white_mat.diffuse_color = (1.0, 1.0, 1.0)
  white_mat.diffuse_intensity = 1.0
  white_mat.specular_intensity = 0.0
  new_ground.data.materials.append(white_mat)
  
  nodes.clear()
  
  bg_node = nodes.new(type='ShaderNodeBackground')
  out_node = nodes.new(type='ShaderNodeOutputWorld')
  
  bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
  bg_node.inputs['Strength'].default_value = 1.0
  
  links.new(bg_node.outputs['Background'], out_node.inputs['Surface'])
  world.use_nodes = True

  # Load materials
  utils.load_materials(args.material_dir)

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
  render_args = bpy.context.scene.render
  render_args.engine = "CYCLES"
  render_args.filepath = output_image
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100
  render_args.tile_x = args.render_tile_size
  render_args.tile_y = args.render_tile_size
  if args.use_gpu == 1:
    # Blender changed the API for enabling CUDA at some point
    if bpy.app.version < (2, 78, 0):
      bpy.context.user_preferences.system.compute_device_type = 'CUDA'
      bpy.context.user_preferences.system.compute_device = 'CUDA_0'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific stuff
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=100)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Figure out the left, up, and behind directions along the plane and record
  # them in the scene structure
  camera = bpy.data.objects['Camera']
  
  r = 12.0
  # set the camera at correct position, determined earlier
  random.seed(output_index * 1000 + view_idx)
  camera.location[0] = r * math.cos(azimuth)
  camera.location[1] = r * math.sin(azimuth)
  height_Y = random.uniform(1, 4.0)
  camera.location[2] = height_Y 
  
  look_at(camera, Vector((0, 0, 0)))
  
  bpy.context.scene.update()  # need to update the scene for camera changes to take effect
  
  plane_normal = plane.data.vertices[0].normal
  cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
  cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
  cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  # Delete the plane; we only used it for normals anyway. The base scene file
  # contains the actual ground plane.
  utils.delete_object(plane)

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)
  
  scene_struct['camera_position'] = tuple(camera.location)

  # Add random jitter to lamp positions
  random.seed(output_index)  # ensure that jitter is consistent across views of the same scene
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects = add_fixed_objects(scene_struct, fixed_objects, args, camera)

  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f, indent=2)

  # Record the azimuth and tilt values used for this image
  if camera_info_path is not None:
    tilt = math.atan2(height_Y, r)  # elevation angle above the ground plane
    camera_info_entry = {
      'scene_index': output_index,
      'view_index': view_idx,
      'split': output_split,
      'image_filename': os.path.basename(output_image),
      'azimuth_radians': azimuth,
      'azimuth_degrees': math.degrees(azimuth),
      'tilt_radians': tilt,
      'tilt_degrees': math.degrees(tilt),
      'camera_height': height_Y,
      'camera_distance': r,
      'camera_position': tuple(camera.location),
    }
    seen = camera_info_seen if camera_info_seen is not None else set()
    append_camera_info(camera_info_path, camera_info_entry, seen)

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)
    
def add_fixed_objects(scene_struct, fixed_objects, args, camera):
  objects = []
  blender_objects = []
  
  for spec in fixed_objects:
    utils.add_object(args.shape_dir, spec['obj_name'], spec['r'], (spec['x'], spec['y']), theta=spec['theta'])
    obj = bpy.context.object
    blender_objects.append(obj)
    
    utils.add_material(spec['mat_name'], Color=spec['rgba'])
    
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': spec['obj_name_out'],
      'size': spec['size_name'],
      'material': spec['mat_name_out'],
      '3d_coords': tuple(obj.location),
      'rotation': spec['theta'],
      'pixel_coords': pixel_coords,
      'color': spec['color_name'],
    })
    
    all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
    if not all_visible:
      print("Some objetcs are occluded in views")
      
  return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.
  
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
  """
  Check whether all objects in the scene have some minimum number of visible
  pixels; to accomplish this we assign random (but distinct) colors to all
  objects, and render using no lighting or shading or antialiasing; this
  ensures that each object is just a solid uniform color. We can then count
  the number of pixels of each color in the output image to check the visibility
  of each object.

  Returns True if all objects are visible and False otherwise.
  """
  f, path = tempfile.mkstemp(suffix='.png')
  object_colors = render_shadeless(blender_objects, path=path)
  img = bpy.data.images.load(path)
  p = list(img.pixels)
  color_count = Counter((p[i], p[i+1], p[i+2], p[i+3])
                        for i in range(0, len(p), 4))
  os.remove(path)
  if len(color_count) != len(blender_objects) + 1:
    return False
  for _, count in color_count.most_common():
    if count < min_pixels_per_object:
      return False
  return True


def render_shadeless(blender_objects, path='flat.png'):
  """
  Render a version of the scene with shading disabled and unique materials
  assigned to all objects, and return a set of all colors that should be in the
  rendered image. The image itself is written to path. This is used to ensure
  that all objects will be visible in the final rendered scene.
  """
  render_args = bpy.context.scene.render

  # Cache the render args we are about to clobber
  old_filepath = render_args.filepath
  old_engine = render_args.engine
  old_use_antialiasing = render_args.use_antialiasing

  # Override some render settings to have flat shading
  render_args.filepath = path
  render_args.engine = 'BLENDER_RENDER'
  render_args.use_antialiasing = False

  # Move the lights and ground to layer 2 so they don't render
  utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
  utils.set_layer(bpy.data.objects['Ground'], 2)

  # Add random shadeless materials to all objects
  object_colors = set()
  old_materials = []
  for i, obj in enumerate(blender_objects):
    old_materials.append(obj.data.materials[0])
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % i
    while True:
      r, g, b = [random.random() for _ in range(3)]
      if (r, g, b) not in object_colors: break
    object_colors.add((r, g, b))
    mat.diffuse_color = [r, g, b]
    mat.use_shadeless = True
    obj.data.materials[0] = mat

  # Render the scene
  bpy.ops.render.render(write_still=True)

  # Undo the above; first restore the materials to objects
  for mat, obj in zip(old_materials, blender_objects):
    obj.data.materials[0] = mat

  # Move the lights and ground back to layer 0
  utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
  utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
  utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
  utils.set_layer(bpy.data.objects['Ground'], 0)

  # Set the render settings back to what they were
  render_args.filepath = old_filepath
  render_args.engine = old_engine
  render_args.use_antialiasing = old_use_antialiasing

  return object_colors


if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')