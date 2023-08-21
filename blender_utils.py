"""
This script contains functions used by both 
morph.py and morphbody.py. 
"""
from typing import List, Tuple
import bpy
from time import time
import json
import os
from mathutils import Matrix

# Prepare working directory for loading body texture images
blender_file_dir = os.path.dirname(bpy.data.filepath)
# Prepare usual config parameters
TMP = "tmp"
# TEMPLATE_JSON = os.path.join(blender_file_dir,
#                              f"./{TMP}/original_measurements.json")
RIGSET_JSON = "./blender_files/rigging_setting.json"
with open(RIGSET_JSON, "r") as f:
    # RIGSET contains parameters that requires very frequent changes during testing phase
    # so we just use a json file to store it, to avoid keep CI/CD and wasting github quota.
    RIGSET = json.load(f) 


def clear_scene():
    # bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def calc_ratios(original_dict: dict, target_dict: dict) -> dict:
    """Take a original measurements (which is a template, always fixed for certain body type)
    and a target measurements (which is provided by user)
    we return a dictionary with the same set of keys as original_dict
    but with values as ratios. 
    For example, if the template's body shoulder is 46cm and user's shoulder is 50cm, 
    then the ratio will be 50/46. 
    The ratios are meant to be used during rigging, where we (for example) rig a bone
    of length 46cm to 50cm.
    
    This function is to be used for both body and garments, talking about efficient
    reusable codes huh?

    Args:
        original_dict (dict): Contains original measurements dictionary of a 
            template body / template garment. 
        target_dict (dict): Contains 

    Returns:
        dict: Contains ratio of compared measurements.
    """
    # Target dict should include all keys in original dict
    # while they don't necessary need to have same set of keys
    # for front-end convenience.
    keys = original_dict.keys()
    # Cancel waist for bottom garment
    """
    Waist is generally already used for top-garment, hence technically 
        I have no way of simultaneously using rigging technique to 
        provide waist rigging for both top and bottom garments.
    If you're reading this then it is probably your job to solve this,
        but without solving this the code still runs fine, 
        unless user provide drastically different waist sizes for 
        their top and bottom garments, then you might want to solve this issue.
        (by previous backend developer)
    """
    if "bottom-length" in original_dict.keys():
        del original_dict["waist"]
    # Top/Bottom garment length names adaptation (change top-length to length)
    for side in ["top", "bottom", "full"]:
        key = side + "-length"
        if key in original_dict.keys():
            target_dict[key] = target_dict["length"]
            del target_dict["length"]

    return {key: target_dict[key] / original_dict[key] for key in keys}

def apply_transformation(obj_name: str):
    """Apply all transformation (Loc, Roc, Scale) for an object. 
    This happens in-place and does not return anything.

    Args:
        obj_name (str): The name of the object.
    """
    obj = bpy.data.objects[obj_name]
    mb = obj.matrix_basis
    if hasattr(obj.data, "transform"):
        obj.data.transform(mb)
    for c in obj.children:
        c.matrix_local = mb @ c.matrix_local
    
    # Let object matrix become identity.
    obj.matrix_basis.identity()

def put_texture(obj_name: str, tex_list: List[str]):
    """Insert body textures onto body. 

    Args:
        obj_name (str): The name of the body mesh.
        tex_list (List[str]): List of relative paths to the texture images.
            The list can only be length of 1 (mannequin) or 6 (realistic)
            because mannequin needs only one texture, 
            but realistic model needs 6 textures, which are
            body texture, body normal, body roughness,
            head texture, head normal and head roughness, respectively. 

            This is also the only way to auto-detect whether the developer
                is using mannequin or realistic body model. 

    Raises:
        ValueError: If length of tex_list is not 1 or 6, this function raises
                        ValueError. 
    """
    # Texture path should be relative path
    if len(tex_list) == 1:  # For Body_ver==1, only one texture needed.
        texture = tex_list[0]
        obj = bpy.data.objects[obj_name]

        mat = bpy.data.materials.get("Material")
        if mat is None:
            mat = bpy.data.materials.new(name="Material")
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        # Specify material slot
        materials = obj.material_slots
        # Force material slots to contain a material contain "Color"
        if len(materials) == 1:
            m = materials[0]
        else:
            remove_inds = []
            for i, mat in enumerate(obj.data.materials):
                if "Color" not in mat.name:
                    remove_inds.append(i)
            remove_inds.sort()
            remove_inds.reverse()

            for ind in remove_inds:
                obj.data.materials.pop(index=ind)

        m = materials[0]
        material = m.material
        material.use_nodes = True
        node_tex = material.node_tree.nodes.new('ShaderNodeTexImage')
        node_principled = material.node_tree.nodes["Principled BSDF"]
        image_node = material.node_tree.nodes["Image Texture"]
        # load image to Blender
        image_path = os.path.join(os.getcwd(), texture)
        try:
            image_obj = bpy.data.images.load(image_path, check_existing=True)
        except:
            print("Image not loaded")
            return
        tex = bpy.data.images.get(image_obj.name)
        image_node.image = tex
        # Link nodes
        links = material.node_tree.links
        link = links.new(image_node.outputs["Color"],
                         node_principled.inputs["Base Color"])

    elif len(tex_list) == 6:
        """
            For Body_ver==2, 6 textures needed. 
            [
                tex_body_path, nor_body_path, rou_body_path,
                tex_head_path, nor_head_path, rou_head_path,
            ]
            For body and head, each needs Texture (tex), Normal (nor) 
                and Roughness (rou).
        """
        expect_str_keys = [
            "body",
            "body_normal",
            "body_roughness",
            "head",
            "head_normal",
            "head_roughness",
        ]
        expect_str_dict = {
            key: os.path.join(os.getcwd(), tex_list[ind]) \
                for ind, key in enumerate(expect_str_keys)
        }
        obj = bpy.data.objects[obj_name]
        print("Body name is", obj_name)
        obj_mats = obj.material_slots
        print(f"Material slots of {obj_name}", obj_mats)
        for mat in obj_mats:
            for section in ["body", "head"]:
                if section in mat.material.name:
                    mat.material.use_nodes = True
                    # Inserting textures into this material
                    ### Insert tex
                    image_obj = bpy.data.images.load(expect_str_dict[section],
                                                     check_existing=True)
                    tex = bpy.data.images.get(image_obj.name)
                    image_node = mat.material.node_tree.nodes["Image Texture"]
                    image_node.image = tex
                    ### Create nodes and links shortcut
                    nodes = mat.material.node_tree.nodes
                    links = mat.material.node_tree.links
                    ### Create roughness node
                    node = nodes.new("ShaderNodeTexImage")
                    image_obj = bpy.data.images.load(
                        expect_str_dict[section + "_roughness"],
                        check_existing=True)
                    tex = bpy.data.images.get(image_obj.name)
                    node.image = tex
                    ### Link roughness node
                    links.new(node.outputs["Color"],
                              nodes["Principled BSDF"].inputs["Roughness"])
                    ### Change roughness node colorspace to non-color.
                    node.image.colorspace_settings.name = 'Non-Color'
                    ### Create normal node
                    node = nodes.new("ShaderNodeTexImage")
                    image_obj = bpy.data.images.load(
                        expect_str_dict[section + "_normal"],
                        check_existing=True)
                    tex = bpy.data.images.get(image_obj.name)
                    node.image = tex
                    ### Link normal node
                    links.new(node.outputs["Color"],
                              nodes["Normal Map"].inputs["Color"])
                    ### Change normal node strength from 0 to 1
                    nodes["Normal Map"].inputs["Strength"].default_value = 1
                    ### Change normal node colorspace to non-color.
                    node.image.colorspace_settings.name = 'Non-Color'
                    break
    else:
        raise ValueError("Texture list should contain 1 element (for body_ver=1) " \
                    + "or 6 elements (for body_ver=2).")


def remove_double(name: str) -> None:
    """Remove overlapped vertices of an object's mesh.
    Why do we need this? We frequently import object from gltf (.glb) format,
    where it stores the mesh as isolated triangles, hence normally we will see
    the mesh is connected, but deep down all triangles are isolated. 
    If we did not address this problem, it will affect the rigging and cloth simulation
    process since isolated triangles cannot normally worked as collision object 
    for cloth simulation. 
    By merging mesh vertices by distance, the mesh coming from gltf format is able 
    to become one connected mesh again, ready to perform other actions on it. 

    Args:
        name (str): The name of the object.
    """
    if name == "":
        return
    obj = bpy.data.objects[name]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    bpy.ops.mesh.remove_doubles()
    bpy.ops.object.mode_set(mode='OBJECT')


def clear_sharp(name: str):
    """Clear sharp edges of object (usually on garments). 
    This probably won't have much uses as it technically does not affect cloth simulation result,
    but I'm too afraid to remove it. 

    Args:
        name (str): The name of the object.
    """
    if name == "": return
    obj = bpy.data.objects[name]
    for edge in obj.data.edges:
        edge.use_edge_sharp = False


def recalculate_body(body_name: str):
    """Recalculate normal of a body mesh. 
    Why we need this function? 
    At the early stage of developing this program we are using mannequin body, 
    both hands of the body are handcrafted, then manually stitched to the body.
    Due to how inexperienced I was, I didn't check the normal of the body and
    it turns out the normal are important for garment sewing and rigging process. 
    Therefore, this function is to recalculate them to make the normals aligned 
    throughout the whole body model. 

    Nowadays we are using realistic body model, which is designed by my another 
    skillful 3D designer colleague, the entire body normals are already aligned well,
    but I still choose to remain this function in the pipeline just to make 
    the entire process more robust.  

    Args:
        body_name (str): The name of the body object.
    """
    body = bpy.data.objects[body_name]
    bpy.ops.object.select_all(action='DESELECT')
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')


def unsymmetrize(rig_name: str):
    """Force the body rig to be not-symmetrized. 
    This is to cater for more real-life body posing as they are often times
    not symmetric. 

    Args:
        rig_name (str): Name of the rig object.
    """
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = bpy.data.objects[rig_name]
    bpy.ops.object.mode_set(mode='POSE')
    bpy.context.object.pose.use_mirror_x = False
    bpy.ops.object.mode_set(mode='OBJECT')


def get_data(ratios: dict) -> tuple[list[str], list[list[tuple, str, tuple]]]:
    """Use ratios obtained from calc_ratios and do a key mapping to bone names.
    Also stores necessary transform for each bone.  

    Args:
        ratios (dict): Key-value pairs of parts w.r.t. its ratio.

    Returns:
        tuple[list[str], list[list[tuple, str, tuple]]]: 
            The first list is simply list of strings, which are the bone names.
            The second list is a list of the form
                [(sx, sy, sz), world_orientation, (flagx, flagy, flagz)].
            The first tuple of floats are the scaling numbers, usually derived from ratios.
            The world_orientation is either "GLOBAL" or "LOCAL", indicating which coordinate 
                system to be used for transformation.
            The last tuple of flags is to restrict the axes of transformation.
            Read the code below to grasp how to use them in practice. 
    """
    bone_names = []
    transforms = []
    for key, ratio in ratios.items():
        if key in ["height", "full-length"]:
            bones = ["spine.001", "thigh.L", "thigh.R"]
            trans = [[(1, 1, ratio), 'GLOBAL', (False, False, True)],
                     [(1, 1, ratio), 'GLOBAL', (False, False, True)],
                     [(1, 1, ratio), 'GLOBAL', (False, False, True)]]
        elif key == "shoulder":
            bones = ["shoulder.L", "shoulder.R"]
            trans = [[(1, ratio, 1), 'GLOBAL', (False, True, False)],
                     [(1, ratio, 1), 'GLOBAL', (False, True, False)]]
        elif key == "chest":
            bones = ["chest.L", "chest.R"]
            trans = [[((3 + ratio) / 4, ratio, 1), 'GLOBAL',
                      (True, True, False)],
                     [((3 + ratio) / 4, ratio, 1), 'GLOBAL',
                      (True, True, False)]]
        elif key == "bust":
            bones = ["bust.L", "bust.R"]
            trans = [[((3 + ratio) / 4, ratio, 1), 'GLOBAL',
                      (True, True, False)],
                     [((3 + ratio) / 4, ratio, 1), 'GLOBAL',
                      (True, True, False)]]
        elif key == "waist":
            bones = ["waist.L", "waist.R"]
            trans = [[((1 + ratio) / 2, ratio, 1), 'GLOBAL',
                      (True, True, False)],
                     [((1 + ratio) / 2, ratio, 1), 'GLOBAL',
                      (True, True, False)]]
        elif key == "bicep":
            bones = [
                "upper_arm.L", "forearm.L", "hand.L", "upper_arm.R",
                "forearm.R", "hand.R"
            ]
            trans = [[(ratio, 1, ratio), 'LOCAL', (True, False, True)],
                     [((1 / ratio + 1) / 2, 1, (1 / ratio + 1) / 2), 'LOCAL',
                      (True, False, True)],
                     [(1 / ratio, 1 / ratio, 1 / ratio), 'GLOBAL',
                      (True, True, True)],
                     [(ratio, 1, ratio), 'LOCAL', (True, False, True)],
                     [((1 / ratio + 1) / 2, 1, (1 / ratio + 1) / 2), 'LOCAL',
                      (True, False, True)],
                     [(1 / ratio, 1 / ratio, 1 / ratio), 'GLOBAL',
                      (True, True, True)]]
        elif key == "sleeve":
            bones = ["upper_arm.L", "upper_arm.R"]
            trans = [[(1, ratio, 1), 'LOCAL', (False, True, False)],
                     [(1, ratio, 1), 'LOCAL', (False, True, False)]]
        elif key == "thigh":
            bones = ["thigh.L", "thigh.R"]
            trans = [[(ratio, ratio, 1), 'GLOBAL', (True, True, False)],
                     [(ratio, ratio, 1), 'GLOBAL', (True, True, False)]]
        elif key == "top-length":
            bones = ["spine.001"]
            trans = [[(1, 1, ratio), 'GLOBAL', (False, False, True)]]
        elif key == "bottom-length":
            bones = ["thigh.L", "thigh.R"]
            trans = [[(1, 1, ratio), 'GLOBAL', (False, False, True)],
                     [(1, 1, ratio), 'GLOBAL', (False, False, True)]]
        elif key == "hip":
            bones = ["hip.L", "butt.L", "hip.R", "butt.R"]
            trans = [[(1, ratio, 1), 'GLOBAL', (False, True, False)],
                     [((1 + ratio) / 2, 1, 1), 'GLOBAL', (True, False, False)],
                     [(1, ratio, 1), 'GLOBAL', (False, True, False)],
                     [((1 + ratio) / 2, 1, 1), 'GLOBAL', (True, False, False)]]
        else:
            continue
        bone_names += bones
        transforms += trans
    return bone_names, transforms


def make_highpoly(name: str):
    """Apply subdivision modifier to make mesh more refined.
    This can be used to improve the visual result of a mesh.

    Args:
        name (str): Name of the mesh object.
    """
    if name == "":
        return
    obj = bpy.data.objects[name]
    subd = obj.modifiers.new("Subdivision", 'SUBSURF')
    subd.levels = RIGSET["body_subd_level"]


def apply_mods(name: str):
    """Apply whatever modifiers of a mesh object following its order of modifiers. 
    Collision is not applied. I don't know what is the effect of applying it 
    so I just skip it. 

    Why do you need this function, you might ask? Because whatever modifier I added before,
    only exists in the blend file. The mesh structure itself does not really affected by 
    the modifier before you apply it. 
    If you want it to have effect exporting as glb file, you'll have to apply the modifier. 

    Args:
        name (str): Name of the object mesh.
    """
    if name == "":
        return
    obj = bpy.data.objects[name]
    bpy.context.view_layer.objects.active = obj
    for mod in obj.modifiers:
        if mod.name != 'Collision':
            bpy.ops.object.modifier_apply(modifier=mod.name)

def ratio_rigging(rig_name: str, bone_names: list[str], transforms: list[list[tuple, str, tuple]]):
    """Rig all bones provided by the user with their transforms specified. 

    Args:
        rig_name (str): The name of the whole rig object.
        bone_names (list[str]): The bone names, can be obtained from get_data.
        transforms (list[list[tuple, str, tuple]]): The list fo transforms, 
                                                can be obtained from get_data.
    """
    bpy.ops.object.select_all(action='DESELECT')
    armature = bpy.data.objects[rig_name]
    bpy.context.view_layer.objects.active = armature
    bones = [armature.data.bones[name] for name in bone_names]
    bpy.ops.object.mode_set(mode='POSE')
    # Loop through bone names
    for i in range(len(bone_names)):
        bpy.ops.pose.select_all(action='DESELECT')
        bones[i].select = True
        if transforms[i][1] == "GLOBAL":
            # Use identity matrix for global transformation
            matrix = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        else:
            # Use Pose Bone's orientation matrix for local transformation.
            # The pose-bone matrix provided by Blender is a 4x4 3D affine transformation matrix
            # (See https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html#:~:text=The%204%20by%204%20transformation,in%20the%20first%20three%20columns.
            # for more details)
            # However, we only need its rotation matrix, composed of its first 3 rows and first 3 columns.
            # Furthermore, the bpy ops function accept orientation matrix in the form of row-based matrix,
            # while pose bone's orientation matrix is column-base, hence the transpose is needed.
            
            # How to understand the orientation matrix? 
            # TODO: Put this information in README
            matrix = armature.pose.bones[bone_names[i]].matrix
            matrix = matrix.transposed()
            matrix = [row[:3] for row in matrix][:3]
        bpy.ops.transform.resize(value=transforms[i][0],
                                 orient_type=transforms[i][1],
                                 orient_matrix=matrix,
                                 orient_matrix_type=transforms[i][1],
                                 constraint_axis=transforms[i][2],
                                 mirror=True,
                                 use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH',
                                 proportional_size=1,
                                 use_proportional_connected=False,
                                 use_proportional_projected=False)

    bpy.ops.object.mode_set(mode='OBJECT')



def post_edit(rig_name: str, gender: str, body_type: str):
    """Manual post edit to fix unnatural rigging result.

    Args:
        rig_name (str): Name of the rig.
        gender (str): Gender, only "male" or "female"
        body_type (str): Body type. If gender is "male" then body type can be
                    "skinny", "normal", or "fat".
                    If gender is "female" then body type can only be 
                    "normal" or "fat".
    """
    if gender == "female":
        return
    HAND_RATIO = {
        tuple(item["key"]): item["value"] for item in RIGSET["hand_ratios"]
    }
    # Adjust general rig size based on real-life visual
    bone_names = ["upper_arm.L", "upper_arm.R", "shoulder.L", "shoulder.R", "hand.L", "hand.R"]
    transforms = [
                    [(1, 0.82, 1), 'LOCAL', (False, True, False)],
                    [(1, 0.82, 1), 'LOCAL', (False, True, False)],
                    [(1, 0.90, 1), 'GLOBAL', (False, True, False)],
                    [(1, 0.90, 1), 'GLOBAL', (False, True, False)],
                    [(1, HAND_RATIO[(gender, body_type)], 1), 'LOCAL', (False, True, False)],
                    [(1, HAND_RATIO[(gender, body_type)], 1), 'LOCAL', (False, True, False)],
                ]
    ratio_rigging(rig_name, bone_names, transforms)



def apply_rig_modifier(name, rig_name="metarig"):
    """Apply rig transforms to a mesh.

    Args:
        name (_type_): Name of the mesh (usually body or garment)
        rig_name (str, optional): Name of the whole rig. Defaults to "metarig".
    """
    if name == "":
        return
    obj = bpy.data.objects[name]
    bpy.context.view_layer.objects.active = obj
    for mod in obj.modifiers:
        if mod.type == 'ARMATURE':
            bpy.ops.object.modifier_apply(modifier=mod.name)
            print(f"Applied Armature modifier for object {name}.")


def clear_parent(name: str):
    """Isolate a mesh (body or garment) from its rig while keeping transformation.

    Args:
        name (str): Name of a mesh object (usually body or garment).
    """
    if name == "":
        return
    obj = bpy.data.objects[name]
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')


def remove_armature(name: str):
    """Delete armature from scene. You should run clear_parent before this,
    or the result would be unexpected. 

    Args:
        name (str): Name of the object (usually the whole rig).
    """
    objs = bpy.data.objects
    objs.remove(objs[name], do_unlink=True)


def export_object(name: str, path: str):
    """Export object to gltf (.glb) format.
    As far as I know, the path do not need to be absolute path.
    Relative path could also work. 

    Args:
        name (str): Name of the object to be exported.
        path (str): Destination path of the object (ends with .glb)
    """
    if name == "":
        return
    obj = bpy.data.objects[name]
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=path,
                              use_selection=True,
                              export_colors=False)


def decimate(name: str, ratio: float = 0.5):
    """Bring down polycount of a mesh, similar to blurring a high-res image. 
    Can be considered inverse of make_highpoly. 

    Args:
        name (str): Name of the object.
        ratio (float, optional): A float between 0 and 1, the more closer to 1 
                    the more quality preserved. Defaults to 0.5.
    """
    if name == "": 
        return
    obj = bpy.data.objects[name]
    # Add modifier
    mod = obj.modifiers.new('DecimateMod', 'DECIMATE')
    mod.ratio = ratio
    mod.use_collapse_triangulate = True
    # Apply modifier
    for mod in obj.modifiers:
        if mod.type == 'DECIMATE':
            bpy.ops.object.modifier_apply(modifier=mod.name)
            print(f"Applied Decimate modifier for object {name}.")
            break


def duplicate_obj(name: str) -> str:
    """Duplicate object on the exact same location. 

    Args:
        name (str): Name of the object.

    Returns:
        str: Name of the duplicated object.
    """
    ori_obj = bpy.data.objects[name]
    new_obj = ori_obj.copy()
    new_obj.data = ori_obj.data.copy()
    new_obj.animation_data_clear()
    bpy.context.collection.objects.link(new_obj)
    return new_obj.name


def delete_obj(name: str):
    """Delete object from the scene.

    Args:
        name (str): _description_
    """
    objs = bpy.data.objects
    objs.remove(objs[name], do_unlink=True)


def find_neck(body_name: str):
    """Find neck location (x, y, z) of the body.

    Args:
        body_name (str): Body name.

    Returns:
        Vector: 3 dimensional vector representing location of the back of the neck. 
            It is neither a list nor a tuple, but you can transform it by simply write
            list(ret) or tuple(ret).
    """
    body = bpy.data.objects[body_name]
    # Get original vertices count
    ori_verts_count = len(body.data.vertices)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.select_all(action='DESELECT')
    body.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.context.tool_settings.mesh_select_mode = (False, True, False)
    bpy.ops.mesh.select_all(action='SELECT')
    # Bisect it
    bpy.ops.mesh.bisect(plane_co=(0, 0, 0),
                        plane_no=(0, 1, 0),
                        flip=False,
                        threshold=0)

    # Find vertices with y=0
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    bpy.ops.object.mode_set(mode='OBJECT')
    vertices_all = body.data.vertices
    bisect_verts_count = len(vertices_all)
    ### Bisect only introduce new indices, it won't insert indices between originals.
    on_y0_indices = list(range(ori_verts_count, bisect_verts_count))

    # Discard indices which occur on top 10% and bottom 50%.
    # Top 15% is around the head, bottom 50% is below chest.
    z_list = [(body.matrix_world @ vertices_all[ind].co)[2]
              for ind in on_y0_indices]
    z_max = max(z_list)
    z_min = min(z_list)
    z_10 = z_min + 0.9 * (z_max - z_min)
    z_50 = z_min + 0.5 * (z_max - z_min)
    on_neck_indices = [
        ind for ind in on_y0_indices
        if abs((body.matrix_world @ vertices_all[ind].co)[2] -
               (z_10 + z_50) / 2) < (z_10 - z_50) / 2
    ]

    # Preserve only indices with negative x (back of the neck)
    on_back_neck_indices = [
        ind for ind in on_neck_indices if vertices_all[ind].co[0] < 0
    ]

    bpy.ops.object.mode_set(mode='EDIT')
    for i, vert in enumerate(vertices_all):
        if i in on_back_neck_indices:
            vert.select = True
        else:
            vert.select = False
    # Find the index with maximum x coordinate
    back_neck_dict = {
        ind: vertices_all[ind].co[0]
        for ind in on_back_neck_indices
    }
    print(back_neck_dict)
    max_ind = max(back_neck_dict, key=back_neck_dict.get)
    print(max_ind)
    bpy.ops.object.mode_set(mode='OBJECT')
    # Get the location then return it
    return body.matrix_world @ vertices_all[max_ind].co


def export_neck_loc_size(body_name: str, body_export_path: str):
    """Find the back of neck's location and its lateral neck diameter from left to right

    Args:
        body_name (str): Name of the body mesh in blender
        body_export_path (str): The location of export body path.
                            We only use this, replace its extension to .json,
                            then use this json path as export path. 
    """
    # Duplicate the body
    body2_name = duplicate_obj(body_name)
    # Location of back of the neck
    neck_loc = list(find_neck(body2_name))
    # Find neck lateral diameter
    body = bpy.data.objects[body2_name]
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.context.tool_settings.mesh_select_mode = (True, False, False
                                                  )  # Vertex mode
    bpy.ops.mesh.select_all(action='SELECT')
    # Bisect it
    bpy.ops.mesh.bisect(plane_co=tuple(neck_loc),
                        plane_no=(0.4, 0, 1),
                        flip=False,
                        threshold=0)
    bpy.ops.object.mode_set(mode="OBJECT")
    # Get the list of bisected contour, then find min max y values
    selected_v = [(body.matrix_world @ v.co)[1] for v in body.data.vertices
                  if v.select]
    ymin = min(selected_v)
    ymax = max(selected_v)
    lateral_diam = ymax - ymin
    # Delete the duplicated body
    delete_obj(body2_name)
    # Export neck data.
    neck_file = os.path.splitext(body_export_path)[0] + ".json"
    with open(neck_file, "w") as f:
        json.dump({"neck_loc": list(neck_loc), "neck_size": lateral_diam}, f)

