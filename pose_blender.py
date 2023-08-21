import sys
import bpy
from mathutils import Vector, Matrix
from math import pi, cos, acos
import os
import numpy as np
import argparse
from typing import Literal
from copy import deepcopy
DIR = os.path.dirname(bpy.data.filepath)
if not DIR in sys.path:
    sys.path.append(DIR)
from blender_utils import *
from const import *

BLENDER_DIR = "./blender_files/"
BLENDER_MALE = os.path.join(BLENDER_DIR, "male/")

def import_files(rigged_body_path: str) -> list[str, str]:
    """Import files for rigging. 

    Args:
        rigged_body_path (str): Path to rigged body. Must be in fbx format.
        garment_path (str): Path to garment with top/full placement. Can be obj or glb.
        pant_path (str): Path to garment with bottom placement. Can be obj or glb.

    Returns:
        list[str, str, str, str]: Names of imported objects,
                    in order: body_name, armature_name, top_garment_name, bottom_garment_name.
    """
    # Import rigged body
    bpy.ops.import_scene.fbx(filepath=rigged_body_path, ignore_leaf_bones = True)
    names = [ob.name for ob in bpy.data.objects]
    types = [ob.type for ob in bpy.data.objects]
    # Get body name
    for i, name in enumerate(names):
        if types[i] == "MESH":
            body_name = name
        elif types[i] == "ARMATURE":
            rig_name = name

    print(f"Imported body object with name: {body_name}")
    print(f"Imported armature object with name: {rig_name}")
    
    return body_name, rig_name

def apply_body_textures(body_name):
    tex_body_path = os.path.join(BLENDER_MALE, "chinese_body.png")
    nor_body_path = os.path.join(BLENDER_MALE, "body_normal.png")
    rou_body_path = os.path.join(BLENDER_MALE, "body_roughness.png")
    tex_head_path = os.path.join(BLENDER_MALE, "chinese_head.png")
    nor_head_path = os.path.join(BLENDER_MALE, "head_normal.png")
    rou_head_path = os.path.join(BLENDER_MALE, "head_roughness.png")
    six_textures = [
        tex_body_path,
        nor_body_path,
        rou_body_path,
        tex_head_path,
        nor_head_path,
        rou_head_path,
    ]
    put_texture(body_name, six_textures)

def set_current_as_rest(rig_name):
    armature = bpy.data.objects[rig_name]
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    bpy.ops.pose.armature_apply(selected=False)    
    bpy.ops.object.mode_set(mode='OBJECT')

def empty_parent(body_name, rig_name):
    body = bpy.data.objects[body_name]
    rig = bpy.data.objects[rig_name]
    bpy.ops.object.select_all(action='DESELECT')
    body.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type='ARMATURE_NAME')

def get_head_tail(rig_name, bone_name):
    # Return head and tail, in Vector format
    rig_obj = bpy.data.objects[rig_name]
    R = rig_obj.matrix_world.to_3x3()
    R = np.array(R)
    t = rig_obj.matrix_world.translation
    t = np.array(t) 
    db = rig_obj.data.bones[bone_name] # data bone
    return Vector(np.dot(R, np.array(db.head_local)) + t), \
        Vector(np.dot(R, np.array(db.tail_local)) + t)   

def set_as_rest(body_name, rig_name):
    # Assume current in pose mode
    bpy.ops.object.mode_set(mode='OBJECT')
    apply_rig_modifier(body_name, rig_name)
    set_current_as_rest(rig_name)
    empty_parent(body_name, rig_name)
    bpy.ops.object.mode_set(mode='POSE')

def get_indices_from_names(*bone_names):
    return [KP_NAMES.index(bone_name) for bone_name in bone_names]

def adjust_pb_by_rotate_diff(pb, from_vec, to_vec):
    M = (
        Matrix.Translation(pb.head) @
        from_vec.rotation_difference(to_vec).to_matrix().to_4x4() @
        Matrix.Translation(-pb.head)
        )
    pb.matrix = M @ pb.matrix

def guide(body_name, rig_name, bone_name, direction, mode: Literal["rotate", "translate"], base_assist=None):
    rig_obj = bpy.data.objects[rig_name]
    if mode == "rotate":
        # code from 
        # https://blender.stackexchange.com/questions/214844/python-how-to-make-pose-bone-rotation-toward-a-point
        pb = rig_obj.pose.bones[bone_name]
        pb_head, pb_tail = get_head_tail(rig_name, bone_name)
        bv = pb_tail - pb_head
        direction = Vector(direction)
        if base_assist is not None:
            assist_bname, pos = base_assist
            if pos in ["head", "tail"]:
                base_ind = 0 if pos == "head" else 1
                base_point = get_head_tail(rig_name, assist_bname)[base_ind]
            elif pos == "mean":
                one, two = get_head_tail(rig_name, assist_bname)
                base_point = (one+two)/2
            # print("Assist on ", base_point)
            direction.normalize()
            direction *= bv.length
            direction = base_point + direction - pb_head
            # print("Adjusted direction is ", direction)


        rd = bv.rotation_difference(direction)

        M = (
            Matrix.Translation(pb.head) @
            rd.to_matrix().to_4x4() @
            Matrix.Translation(-pb.head)
            )
        pb.matrix = M @ pb.matrix
        
        bpy.ops.object.mode_set(mode='OBJECT')
        apply_rig_modifier(body_name, rig_name)
        set_current_as_rest(rig_name)
        empty_parent(body_name, rig_name)
        bpy.ops.object.mode_set(mode='POSE')

        # print(bone_name, get_head_tail(rig_name, bone_name))

    elif mode == "translate":
        pb = rig_obj.pose.bones[bone_name]
        pb_head = get_head_tail(rig_name, bone_name)[0]
        # bv = pb_tail - pb_head
        direction = Vector(direction)
        if base_assist is None:
            raise ValueError("Within translate mode, assist must be given")
        assist_bname, pos = base_assist
        if pos in ["head", "tail"]:
            base_ind = 0 if pos == "head" else 1
            base_point = get_head_tail(rig_name, assist_bname)[base_ind]
        elif pos == "mean":
            one, two = get_head_tail(rig_name, assist_bname)
            base_point = (one+two)/2
        # print("Assist on ", base_point)
        direction.normalize()
        direction *= (pb_head - base_point).length
        direction = base_point + direction - pb_head
        # rd = bv.rotation_difference(direction)

        M = (
            Matrix.Translation(pb.head) @
            Matrix([[1,0,0,direction[0]],[0,1,0,direction[1]],[0,0,1,direction[2]],[0,0,0,1]]) @
            Matrix.Translation(-pb.head)
        )
        pb.matrix = M @ pb.matrix
        
        bpy.ops.object.mode_set(mode='OBJECT')
        apply_rig_modifier(body_name, rig_name)
        set_current_as_rest(rig_name)
        empty_parent(body_name, rig_name)
        bpy.ops.object.mode_set(mode='POSE')

def cross_product(u, v):
    return Vector((u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]))

def find_orthogonal(x,y,u,v):
    # Assuming x ortho y, u ortho v, all four are unit vectors, type Vector.
    # This return a 4x4 matrix, where the 3x3 part is orthogonal
    # that satisfies Ax = u, Ay = v. 
    # Formula is A=[u,v,w]*[x,y,z]^T 
    # where z=cross(x,y), w=cross(u,v)
    z = cross_product(x,y)
    w = cross_product(u,v)
    left_mat = Matrix((u,v,w))
    left_mat.transpose()
    right_mat = Matrix((x,y,z))
    ret = (left_mat @ right_mat).to_4x4()
    return ret

def adjust_pb_by_orthogonal(pb, x, y, u, v):
    M = (
        Matrix.Translation(pb.head) @
        find_orthogonal(x,y,u,v) @
        Matrix.Translation(-pb.head)
        )
    pb.matrix = M @ pb.matrix

def adjust_pb_by_4x4(pb, mat_4x4):
    M = (
        Matrix.Translation(pb.head) @
        mat_4x4 @
        Matrix.Translation(-pb.head)
        )
    pb.matrix = M @ pb.matrix 

def bound_angle(a, b_norm, b_down):
    # Angle in radian form
    # Calibrate b_down and b_norm into a more natural position
    # Currently works for forearms.
    costheta = - a.dot(b_down)
    cospi6 = cos(pi/6)
    if costheta <= cospi6: # Theta between pi/6 and pi
        return b_norm, b_down
    else: # Theta too small (< pi/6)
        # Force theta to be pi/6.
        print(f"Angle between forearm and upperarm is too small ({acos(costheta)*180/pi} degree). Changed it to 30 degree.")
        c = cross_product(cross_product(a, b_down), a)
        c.normalize()
        new_b_down = 0.5 * c - cospi6 * a
        new_b_norm = - cospi6 * c - 0.5 * a
        return  new_b_norm, new_b_down
    
def spine_get_orthogonal(body_orientations):
    x = body_orientations["all"]["normal"]
    y = body_orientations["all"]["down"]
    u = body_orientations["upper_body"]["normal"]
    v = body_orientations["upper_body"]["down"]
    p = deepcopy((x+u)/2)
    q = deepcopy((y+v)/2)
    p.normalize()
    q = cross_product(p, cross_product(q, p))
    q.normalize()
    first_rot = find_orthogonal(x, y, p, q)
    second_rot = find_orthogonal(p, q, u, v)
    return first_rot, second_rot


def adjust_workflow(body_name, rig_name, kps, calibrate = True):
    rig_obj = bpy.data.objects[rig_name]
    bpy.context.view_layer.objects.active = rig_obj
    bpy.ops.object.mode_set(mode='POSE')
    # Clear transform first
    bpy.ops.pose.transforms_clear()
    # Tracking body position
    body_orientations = {}
    # Adjust overall orientation
    ind1, ind2, ind3 = get_indices_from_names('torso', 'RHip', 'LHip')
    body_normvec = cross_product(Vector(kps[ind3]) - Vector(kps[ind2]),
                                Vector(kps[ind1]) - Vector(kps[ind2]))
    body_downvec = (Vector(kps[ind2])+Vector(kps[ind3]))/2 - Vector(kps[ind1])
    body_normvec.normalize()
    body_downvec.normalize()
    body_orientations["all"] = {"normal": deepcopy(body_normvec), "down": deepcopy(body_downvec)}
    pb = rig_obj.pose.bones["spine"]
    adjust_pb_by_orthogonal(pb, Vector((1,0,0)), Vector((0,0,-1)), body_normvec, body_downvec)
    set_as_rest(body_name, rig_name)
    # Adjust upper_body position
    ind1, ind2, ind3 = get_indices_from_names('neck', 'RShoulder', 'LShoulder')
    chest_normvec = cross_product(Vector(kps[ind3]) - Vector(kps[ind2]),
                                Vector(kps[ind1]) - Vector(kps[ind2]))
    chest_downvec = (Vector(kps[ind2])+Vector(kps[ind3]))/2 - Vector(kps[ind1])
    chest_normvec.normalize()
    chest_downvec.normalize()
    body_orientations["upper_body"] = {"normal": deepcopy(chest_normvec), "down": deepcopy(chest_downvec)}
    first_rot, second_rot = spine_get_orthogonal(body_orientations)
    pb_1 = rig_obj.pose.bones["spine.001"]
    pb_2 = rig_obj.pose.bones["spine.002"]
    adjust_pb_by_4x4(pb_1, first_rot)
    set_as_rest(body_name, rig_name)
    adjust_pb_by_4x4(pb_2, second_rot)
    set_as_rest(body_name, rig_name)
    # Adjust head facing direction
    ind1, ind2, ind3 = get_indices_from_names('nose', 'head', 'neck')
    face_downvec = Vector(kps[ind3]) - Vector(kps[ind2])
    head_to_nose = Vector(kps[ind1]) - Vector(kps[ind2])
    face_normvec = cross_product(cross_product(face_downvec, head_to_nose), face_downvec)
    face_normvec.normalize()
    face_downvec.normalize()
    body_orientations["face"] = {"normal": deepcopy(face_normvec), "down": deepcopy(face_downvec)}
    pb = rig_obj.pose.bones["spine.004"]
    adjust_pb_by_orthogonal(pb, body_orientations["upper_body"]["normal"], body_orientations["upper_body"]["down"],
                                body_orientations["face"]["normal"], body_orientations["face"]["down"])
    set_as_rest(body_name, rig_name)
    # Adjust left and right arms
    for armpos in ["L", "R"]:
        # Upper arms
        ind1, ind2, ind3 = get_indices_from_names(f'{armpos}Shoulder', f'{armpos}Elbow', f'{armpos}Wrist')
        arm_downvec, forearm_downvec = Vector(kps[ind2]) - Vector(kps[ind1]), Vector(kps[ind3]) - Vector(kps[ind2])
        arm_normvec = cross_product(arm_downvec, cross_product(forearm_downvec, arm_downvec))
        arm_normvec.normalize() 
        arm_downvec.normalize()
        body_orientations[f"upper_arm.{armpos}"] = {"normal": deepcopy(arm_normvec), "down": deepcopy(arm_downvec)}
        # Calculate arm down direction from blender
        original_arm_down = get_head_tail(rig_name, f"upper_arm.{armpos}")
        original_arm_down = original_arm_down[1] - original_arm_down[0]
        # Make sure it is orthonormal to upper_body normvec
        original_arm_down = cross_product(cross_product(body_orientations["upper_body"]["normal"], original_arm_down), body_orientations["upper_body"]["normal"])
        original_arm_down.normalize()
        pb = rig_obj.pose.bones[f"upper_arm.{armpos}"]
        adjust_pb_by_orthogonal(pb, body_orientations["upper_body"]["normal"], original_arm_down,
                                body_orientations[f"upper_arm.{armpos}"]["normal"], body_orientations[f"upper_arm.{armpos}"]["down"])
        set_as_rest(body_name, rig_name)
        # Forearms
        forearm_downvec.normalize()
        forearm_normvec = cross_product(cross_product(arm_downvec, forearm_downvec), forearm_downvec)
        forearm_normvec.normalize()
        # Calculate forearm down direction from blender
        original_forearm_down = get_head_tail(rig_name, f"forearm.{armpos}")
        original_forearm_down = original_forearm_down[1] - original_forearm_down[0]
        # Make sure it is orthonormal to upper_arm normvec
        original_forearm_down = cross_product(cross_product(body_orientations[f"upper_arm.{armpos}"]["normal"], original_forearm_down), body_orientations[f"upper_arm.{armpos}"]["normal"])
        original_forearm_down.normalize()
        # Calibrate forearm if needed, make angle between them force to 30deg - 180deg
        if calibrate:
            forearm_normvec, forearm_downvec = bound_angle(original_forearm_down, forearm_normvec, forearm_downvec)
        # Record forearm orientations after optional calibration
        body_orientations[f"forearm.{armpos}"] = {"normal": deepcopy(forearm_normvec), "down": deepcopy(forearm_downvec)}
        pb = rig_obj.pose.bones[f"forearm.{armpos}"]
        adjust_pb_by_orthogonal(pb, body_orientations[f"upper_arm.{armpos}"]["normal"], original_forearm_down,
                                body_orientations[f"forearm.{armpos}"]["normal"], body_orientations[f"forearm.{armpos}"]["down"])
        set_as_rest(body_name, rig_name)
    # Adjust left and right legs
    for legpos in ["L", "R"]:
        # Thighs
        ind1, ind2, ind3 = get_indices_from_names(f'{legpos}Hip', f'{legpos}Knee', f'{legpos}Ankle')
        thigh_downvec, shin_downvec = Vector(kps[ind2]) - Vector(kps[ind1]), Vector(kps[ind3]) - Vector(kps[ind2])
        thigh_normvec = cross_product(cross_product(shin_downvec, thigh_downvec), thigh_downvec)
        thigh_normvec.normalize()
        thigh_downvec.normalize()
        body_orientations[f"thigh.{legpos}"] = {"normal": deepcopy(thigh_normvec), "down": deepcopy(thigh_downvec)}
        # Get thigh downvec in blender, make sure it is orthonormal to "all" normvec
        original_thigh_down = get_head_tail(rig_name, f"thigh.{legpos}")
        original_thigh_down = original_thigh_down[1] - original_thigh_down[0]
        original_thigh_down = cross_product(cross_product(body_orientations["all"]["normal"], original_thigh_down), body_orientations["all"]["normal"])
        original_thigh_down.normalize()
        # Adjusting
        pb = rig_obj.pose.bones[f"thigh.{legpos}"]
        adjust_pb_by_orthogonal(pb, body_orientations["all"]["normal"], original_thigh_down,
                                body_orientations[f"thigh.{legpos}"]["normal"], body_orientations[f"thigh.{legpos}"]["down"])
        set_as_rest(body_name, rig_name)
        # Shins
        shin_normvec = cross_product(cross_product(shin_downvec, thigh_downvec), shin_downvec)
        shin_normvec.normalize()
        shin_downvec.normalize()
        body_orientations[f"shin.{legpos}"] = {"normal": deepcopy(shin_normvec), "down": deepcopy(shin_downvec)}
        # Get shin downvec in blender, make sure it is orthonormal to "thigh.{legpos}" normvec
        original_shin_down = get_head_tail(rig_name, f"shin.{legpos}")
        original_shin_down = original_shin_down[1] - original_shin_down[0]
        original_shin_down = cross_product(cross_product(body_orientations[f"thigh.{legpos}"]["normal"], original_shin_down), body_orientations[f"thigh.{legpos}"]["normal"])
        original_shin_down.normalize()
        # Adjusting
        pb = rig_obj.pose.bones[f"shin.{legpos}"]
        adjust_pb_by_orthogonal(pb, body_orientations[f"thigh.{legpos}"]["normal"], original_shin_down,
                                body_orientations[f"shin.{legpos}"]["normal"], body_orientations[f"shin.{legpos}"]["down"])
        set_as_rest(body_name, rig_name)

    bpy.ops.object.mode_set(mode='OBJECT')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-json", help='Path to the 17 3d-keypoints from MotionBert, format should be .npy.', type=str)
    parser.add_argument('--store-as', type=str, default=None)
    parser.add_argument('--body-model', type=str, default=None, help=f'Select from [{", ".join(list(BODY_MODELS.keys())[1:])}], default to None, which is male model.')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.pose_json is None or not args.pose_json.endswith(".npy"):
        raise ValueError("The format of --pose-json argument is not correct. It is mandatory and should ends with .npy.")
    # Load keypoints (17 points from 3D, as shape [17,3].)
    kps = np.load(args.pose_json)
    assert kps.shape == (17, 3), "Keypoints format incorrect."
    
    clear_scene()
    
    # Import rigged body
    if args.body_model is None:
        rigged_body_path =  "./blender_files/male/chinese_skinned_rigged.fbx"
        body_name, rig_name = import_files(rigged_body_path=rigged_body_path)
        apply_transformation(rig_name)
        # Apply body textures
        apply_body_textures(body_name=body_name)
    else:
        rigged_body_path =  f"./blender_files/{args.body_model}/model.fbx"
        body_name, rig_name = import_files(rigged_body_path=rigged_body_path)
        apply_transformation(rig_name)
    # Remove doubles from imported objects
    remove_double(body_name)
    # Recalculate normal of 3D body due to open hand stitching
    recalculate_body(body_name)
    # X-symmetrize armature
    unsymmetrize(rig_name)
    # Make the current armature as rest pose
    apply_rig_modifier(body_name, rig_name)
    set_current_as_rest(rig_name)
    empty_parent(body_name, rig_name)

    # Adjust bones based on keypoints
    adjust_workflow(body_name = body_name, rig_name=rig_name, kps=kps)

    # Export body
    if args.store_as is not None and args.store_as.endswith(".glb"):
        clear_parent(body_name)
        remove_armature(rig_name)
        export_object(body_name, args.store_as)