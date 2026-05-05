from flask import Flask, render_template, Response
import cv2
import numpy as np
import time
import os
from lwcc import LWCC

app = Flask(__name__)

# Global variables for the model
MODEL_NAME = "DM-Count"
MODEL_WEIGHTS = "QNRF"
model = None

def get_model():
    global model
    if model is None:
        print(f"Loading model {MODEL_NAME} ({MODEL_WEIGHTS})...")
        model = LWCC.load_model(model_name=MODEL_NAME, model_weights=MODEL_WEIGHTS)
    return model

def gen_frames():
    camera = cv2.VideoCapture(0)
    lwcc_model = get_model()
    
    count = 0
    frame_count = 0
    process_every = 10 # Inference every 10 frames
    last_density_color = None
    temp_img_path = "web_temp_frame.jpg"

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_count += 1
            
            # Perform inference periodically
            if frame_count % process_every == 0 or frame_count == 1:
                try:
                    cv2.imwrite(temp_img_path, frame)
                    curr_count, density_map = LWCC.get_count(temp_img_path, model=lwcc_model, return_density=True)
                    count = int(round(curr_count))
                    
                    # Process density map
                    density_norm = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
                    last_density_color = cv2.applyColorMap(density_norm.astype(np.uint8), cv2.COLORMAP_JET)
                    last_density_color = cv2.resize(last_density_color, (frame.shape[1], frame.shape[0]))
                except Exception as e:
                    print(f"Inference error: {e}")

            # Create display (Side-by-side)
            if last_density_color is not None:
                display_frame = np.hstack((frame, last_density_color))
            else:
                display_frame = frame.copy()

            # Add count overlay
            cv2.putText(display_frame, f"Count: {count}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', display_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
