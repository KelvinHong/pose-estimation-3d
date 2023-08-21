import PySimpleGUI as sg
import cv2
import datetime
import os
import subprocess
import glob
from const import BODY_MODELS

CURRENT_IMAGE = None
MODELS_KEYS = list(BODY_MODELS.keys())

def capture_image(save_as):
    # Save as should be a list
    cam = cv2.VideoCapture(0)

    instruction = "Press SpaceBar to capture image"
    cv2.namedWindow(instruction)

    if type(save_as) == str:
        save_as = [save_as]
    total_count = len(save_as)
    image_count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow(instruction, frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            cv2.imwrite(save_as[image_count], frame)
            print("{} written!".format(save_as[image_count]))
            image_count += 1
            if image_count >= total_count:
                break

    cam.release()

    cv2.destroyAllWindows()

layout = [
   [
      sg.Text(text='3D Pose Estimator',
        font=('Arial Bold', 16),
        size=20, expand_x=True,
        justification='center'),
    ],
   [    
        sg.Image(
            expand_x=True, 
            expand_y=True,
            key="IMAGE",
        )
    ],
    [
       sg.Button(
          "Upload Image",
          key='CAPTURE'
       ),
       sg.Combo(MODELS_KEYS, default_value="male", font=('Arial Bold', 14),  expand_x=True, enable_events=True,  readonly=True, key='COMBO'),
       sg.Button(
           "Start 3D Pose Estimation",
           key='ESTIMATE',
       )
    ]
]
window = sg.Window('Application', layout)
while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break
    elif event == 'CAPTURE':
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
        image_path = f"./examples/demo/cap_{timestamp}.png"
        capture_image(image_path)
        if not os.path.isfile(image_path):
            raise RuntimeError("Image is not captured.")
        window['IMAGE'].update(filename=image_path)
        CURRENT_IMAGE = image_path
    elif event == 'ESTIMATE':
        if CURRENT_IMAGE is None:
            sg.popup("Please select an image before proceed.")
            continue
        body_model = values["COMBO"]
        if body_model == "male":
            body_args = []
        else:
            body_args = ["--body-model", body_model]
        subprocess.call(["./venv/Scripts/python.exe", "app.py", "--image", CURRENT_IMAGE] + body_args)
        # Find latest glb
        list_of_files = glob.glob('./examples/demo/*.glb') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        os.startfile(latest_file)
window.close()