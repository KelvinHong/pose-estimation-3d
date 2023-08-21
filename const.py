# Specify all possible body models here
import os
BODY_MODELS = {}
for name in os.listdir("./blender_files/"):
    prefix = "./blender_files/" + name
    if os.path.isdir(prefix):
        BODY_MODELS[name] = name

KP_NAMES = ['root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 
    'LKnee', 'LAnkle', 'torso', 'neck', 'nose', 
    'head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 
    'RElbow', 'RWrist',
]
LINKS = [
    ['head', 'nose', 'neck', 'torso', 'root'],
    ['LWrist', 'LElbow', 'LShoulder', 'neck', 'RShoulder', 'RElbow', 'RWrist'],
    ['LAnkle', 'LKnee', 'LHip', 'root', 'RHip', 'RKnee', 'RAnkle'],
]
# Below are bones that have KP_NAMES as their head
HEAD_OF_BONES = [
    'spine.001', 'thigh.R', 'shin.R', 'foot.R', 'thigh.L',
    'shin.L', 'foot.L', 'spine.003', 'spine.004', '',
    '', 'upper_arm.L', 'forearm.L', 'hand.L', 'upper_arm.R',
    'forearm.R', 'hand.R', 
]
