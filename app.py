# CPU version
import argparse
import os
import platform
import sys
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import imageio
from tqdm import tqdm
import natsort
import subprocess
import cv2
# Constants
from const import *
# AlphaPose libraries
from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter
# MotionBert libraries
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset
# from lib.utils.vismo import render_and_save


parser = argparse.ArgumentParser(description='3DPose Program')
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--body-model', default=None, 
                    help=f'Select from [{", ".join(list(BODY_MODELS.keys())[1:])}], default to None, which is male model.')
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default=None)
parser.add_argument('--sp', default=True, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--posebatch', type=int, default=64,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument("--save_profile_to", default=None, 
                        help="Save compute time into a json file")

INDENT = 4
SPACE = " "
NEWLINE = "\n"
# This program includes software developed by jterrace and David Kim
# in https://stackoverflow.com/questions/10097477/python-json-array-newlines
# Huge thanks to them!
# Changed basestring to str, and dict uses items() instead of iteritems().
def to_json(o, level=0):
    ret = ""
    if isinstance(o, dict):
        ret += NEWLINE + SPACE * INDENT * level + "{" + NEWLINE
        comma = ""
        for k, v in o.items():
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level + 1)
            ret += '"' + str(k) + '":' + SPACE
            ret += to_json(v, level + 1)

        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        ret += '"' + o.strip() + '"'
    elif isinstance(o, list):
        ret += "[" + ",".join([to_json(e, level + 1) for e in o]) + "]"
    # Tuples are interpreted as lists
    elif isinstance(o, tuple):
        ret += "[" + ",".join(to_json(e, level + 1) for e in o) + "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, float):
        ret += '%.7g' % o
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.integer):
        ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.inexact):
        ret += "[" + ','.join(map(lambda x: '%.7g' % x,
                                  o.flatten().tolist())) + "]"
    elif o is None:
        ret += 'null'
    else:
        raise TypeError("Unknown type '%s' for json serialization" %
                        str(type(o)))
    return ret

def pretty_dump(data: dict, filename: str):
    # Make sure parent directory exist.
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    json_string = to_json(data)
    with open(filename, "w") as f:
        f.write(json_string)

def capture_image():
    cap = cv2.VideoCapture(0)

    img_path = 'examples/demo/cap_{}.png'.format(int(time.time()))
    start_time = time.time()
    while True:
        _, img = cap.read()

        cv2.imshow('original', img)

        if cv2.waitKey(1) == ord('q'):
            break
        elif time.time() - start_time > 5:
            cv2.imwrite(img_path, img)
            break

    cap.release()
    cv2.destroyAllWindows()
    if os.path.isfile(img_path):
        return img_path
    else:
        raise SystemError("Image not captured.")

def alphapose(args):
    cfg = update_config("configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml")
    args.inputpath, input_source = os.path.split(args.inputimg)
    input_source = [input_source]
    # Load detection loader
    det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode="image", queueSize=1024)
    # det_worker = det_loader.start()
    det_loader.start()

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    # pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    pose_model.to(args.device)
    pose_model.eval()

    queueSize = 1024
    writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize).start()
    data_len = det_loader.length
    # im_names_desc = range(data_len)
    batchSize = args.posebatch
    # Start 2D pose estimation
    with torch.no_grad():
        (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
        if orig_img is None:
            raise ValueError("No image detected.")
        if boxes is None or boxes.nelement() == 0:
            writer.save(None, None, None, None, None, orig_img, im_name)
            raise ValueError("No Bounding Boxes detected")
        # Pose Estimation
        inps = inps.to(args.device)
        datalen = inps.size(0)
        leftover = 0
        if (datalen) % batchSize:
            leftover = 1
        num_batches = datalen // batchSize + leftover
        hm = []
        for j in range(num_batches):
            inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
            hm_j = pose_model(inps_j)
            hm.append(hm_j)
        hm = torch.cat(hm)
        hm = hm.cpu()
        writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
    # Finishing sequence
    while(writer.running()):
        time.sleep(1)
        print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
    writer.stop()
    det_loader.stop()
    print("===2D Pose Estimation Done, proceed to pose lifting to 3D.===")

def motionbert(args):
    config_path = "configs/pose3d/MB_ft_h36m_global_lite.yaml"
    bert_args = get_config(config_path)

    model_backbone = load_backbone(bert_args)

    model_backbone = nn.DataParallel(model_backbone)
    if torch.cuda.is_available():
        model_backbone = model_backbone.cuda()

    cpt_path = 'checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin'
    print('Loading checkpoint', cpt_path)
    checkpoint = torch.load(cpt_path, map_location=lambda storage, loc: storage)
    # print(checkpoint['model_pos'])
    model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    model_pos = model_backbone
    model_pos.eval()
    testloader_params = {
          'batch_size': 1,
          'shuffle': False,
          'num_workers': 0,
          'pin_memory': True,
          'prefetch_factor': None,
        #   'persistent_workers': True,
          'drop_last': False
    }

    # vid = imageio.get_reader(opts.vid_path,  'ffmpeg')
    # fps_in = vid.get_meta_data()['fps']
    # vid_size = vid.get_meta_data()['size']
    # os.makedirs(opts.out_path, exist_ok=True)

    alphapose_jsonpath = os.path.join(args.outputpath, "alphapose-results.json")
    wild_dataset = WildDetDataset(alphapose_jsonpath, clip_len=243, scale_range=[1,1], focus=None)

    test_loader = DataLoader(wild_dataset, **testloader_params)

    with torch.no_grad():
        batch_input = next(iter(test_loader))
        # for batch_input in tqdm(test_loader):
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
        if bert_args.no_conf:
            batch_input = batch_input[:, :, :, :2]
        if bert_args.flip:    
            batch_input_flip = flip_data(batch_input)
            predicted_3d_pos_1 = model_pos(batch_input)
            predicted_3d_pos_flip = model_pos(batch_input_flip)
            predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
            predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
        else:
            predicted_3d_pos = model_pos(batch_input)
        if bert_args.rootrel:
            predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
        else:
            predicted_3d_pos[:,0,0,2]=0
            pass
        if bert_args.gt_2d:
            predicted_3d_pos[...,:2] = batch_input[...,:2]
        ret = predicted_3d_pos.cpu().numpy()
        # results_all.append(predicted_3d_pos.cpu().numpy())
    print("===3D Pose Lifting Inference done. Proceed to Blender Application.===")
    return ret

def visualize(kps):
    
    import matplotlib
    matplotlib.use("qt5Agg", force=True)
    from matplotlib import pyplot as plt
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    # Plot keypoints
    for kp in kps:
        ax.scatter(kp[0], kp[1], kp[2], c='red', marker='*', s=10)
    # Plot segments
    for link in LINKS:
        for name1, name2 in zip(link[:-1], link[1:]):
            ind1 = KP_NAMES.index(name1)
            ind2 = KP_NAMES.index(name2)
            ax.plot3D(
                [kps[ind1][0],kps[ind2][0]],
                [kps[ind1][1],kps[ind2][1]],
                [kps[ind1][2],kps[ind2][2]],
                color = 'g'
            )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    # plt.savefig("tmp.png")

def main():
    # Parse arguments
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.gpus = [-1]
    args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'
    args.checkpoint = "pretrained_models/halpe26_fast_res50_256x192.pth"
    if platform.system() == 'Windows':
        args.sp = True
    if not args.inputimg:
        # Use camera to capture
        args.inputimg = capture_image()
    if args.outputpath is None:
        # Use same folder as input image
        args.outputpath = os.path.dirname(args.inputimg)
    # Configs
    performance = {}
    # Extract 2D pose
    start = time.time()
    alphapose(args)
    performance["alphapose"] = time.time() - start

    # Lift 2D pose to 3D pose
    start = time.time()
    lifted_3d_kps = motionbert(args)
    performance["motionbert"] = time.time() - start

    # Remove alphapose json file
    os.remove(os.path.join(args.outputpath, "alphapose-results.json"))

    # Visualize on Matplotlib
    cleaned_kps = lifted_3d_kps[0,0] # Flatten and revert scale
    cleaned_kps[:, 1:] *= -1 # Flip y and z axes
    cleaned_kps[:, [0,1,2]] = cleaned_kps[:, [2,0,1]] # [x,y,z] -> [y,z,x]
    # for i, kps in enumerate(cleaned_kps):
    #     print( kps, KP_NAMES[i])

    # Store keypoints into npy file, before sending to Blender
    kps_path = os.path.join(args.outputpath, "pose_3d.npy")
    np.save(kps_path, cleaned_kps)

    # visualize(cleaned_kps)
    # subprocess.call(["blender", "empty.blend", "--python", "pose_blender.py", "--", "--pose-json", "examples\demo\pose_3d.npy"])
    start = time.time()
    body_model_arg = []
    if args.body_model is not None:
        body_model_arg = ["--body-model", args.body_model]
    subprocess.call(["blender", 
                     "-b",
                    "empty.blend", 
                     "--python", 
                     "pose_blender.py", "--", "--pose-json", kps_path,
                     "--store-as", os.path.splitext(args.inputimg)[0] + "_pose.glb"] \
                     + body_model_arg)
    end = time.time()
    performance["blender"] = time.time() - start
    os.remove(kps_path)

    # Save 
    if args.save_profile_to is not None:
        # Assume path contains an empty array or an array of existing performances. 
        # File must already been exist.
        with open(args.save_profile_to, "r") as f:
            performances = json.load(f)
        performances.append(performance)
        pretty_dump(performances, args.save_profile_to)
        
if __name__ == "__main__":
    main()