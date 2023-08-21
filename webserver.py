from flask import Flask, render_template, request, send_from_directory
import os
from werkzeug.utils import secure_filename
import subprocess
import time
import json
from threading import Timer
import webbrowser
from const import  *
 
#*** Backend operation
 
# WSGI Application
# Defining upload folder path
UPLOAD_FOLDER = os.path.join('web_files', 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Clear previously stored images
for file in os.listdir(UPLOAD_FOLDER):
    os.remove(os.path.join(UPLOAD_FOLDER, file))
# Create folder for storing time used for creating 3D files
PERFORMANCE_FILE = os.path.join('web_files', 'performance.json')
if not os.path.isfile(PERFORMANCE_FILE):
    with open(PERFORMANCE_FILE, "w") as f:
        json.dump([], f)
# Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name for template path
# The default folder name for static files should be "static" else need to mention custom folder for static path
app = Flask(__name__, 
            template_folder=os.path.join("web_files", "templates"),
            static_folder=os.path.join("web_files", "static"),
        )
# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
def get_avg_time():
    with open(PERFORMANCE_FILE, "r") as f:
        performances = json.load(f)
    if not performances:
        return "unknown" # No record
    last3_times = [sum(t.values()) for t in performances[-3:]]
    return str(round(sum(last3_times)/len(last3_times), 3)) + "s"

@app.route('/')
def index():
   return render_template("accept_image.html", avg_time= get_avg_time(), 
                          models=[{"code": code, "desc": desc} for code, desc in BODY_MODELS.items()])

@app.route('/models/<path:path>')
def send_model(path):
    return send_from_directory(UPLOAD_FOLDER, path)

@app.route('/images/<path:path>')
def send_image(path):
    return send_from_directory(UPLOAD_FOLDER, path)

@app.route('/upload',  methods=["POST"])
def uploadFile():
    if request.method == 'POST':
        # Upload file flask
        uploaded_img = request.files['pose-file']
        body_model = request.form["bodymodel"]
        
        headless = False if "headless" not in request.form else request.form["headless"] # A flag. If true, only return file path.

        body_model_arg = [] if body_model == "male" else ["--body-model", body_model]
        # Extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)
        # Upload file to database (defined uploaded folder in static path)
        img_staticpath = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        uploaded_img.save(img_staticpath)

        # Do something
        print("Running 3D pose model...")
        start = time.time()
        subprocess.call([
            "venv/Scripts/python.exe", "app.py", 
            "--image", img_staticpath,
            "--save_profile_to", PERFORMANCE_FILE,
        ] + body_model_arg)
        delta = time.time() - start
        print(f"Use time {round(delta, 2)}s. ")

        # Render glb file
        generated_glb = os.path.basename(img_staticpath)
        generated_glb = os.path.splitext(generated_glb)[0] + "_pose.glb"
        if headless:
            return {
                "path_to_img": f"/images/{os.path.basename(img_staticpath)}", 
                "path_to_glb": f"/models/{generated_glb}",
            }
        else:
            return render_template("show_3dpose.html", 
                                path_to_img = f"/images/{os.path.basename(img_staticpath)}", 
                                path_to_glb = f"/models/{generated_glb}"
                            )

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
      Timer(1, open_browser).start()
      app.run()