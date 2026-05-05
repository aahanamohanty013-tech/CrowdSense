import cv2
import argparse
import time
import numpy as np
from lwcc import LWCC

def realtime_counting(model_name="DM-Count", model_weights="SHB", source=0, show_density=False):
    """
    Captures video from webcam or file and performs real-time crowd counting.
    """
    print(f"Loading model {model_name} with weights {model_weights}...")
    model = LWCC.load_model(model_name=model_name, model_weights=model_weights)
    
    # Initialize capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print("Starting real-time capture. Press 'q' to quit.")
    
    count = 0.0
    display_frame = None
    frame_count = 0
    process_every = 15 # Process every 15 frames to maintain good video FPS
    
    temp_img_path = "temp_frame.jpg"
    last_density_color = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        start_time = time.time()
        
        # Only perform inference every N frames
        if frame_count % process_every == 0 or frame_count == 1:
            try:
                # Save frame to temp file as LWCC expects a path
                cv2.imwrite(temp_img_path, frame)
                
                if show_density:
                    count, density_map = LWCC.get_count(temp_img_path, model=model, return_density=True)
                    
                    # Normalize and colorize density map
                    density_norm = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
                    last_density_color = cv2.applyColorMap(density_norm.astype(np.uint8), cv2.COLORMAP_JET)
                    last_density_color = cv2.resize(last_density_color, (frame.shape[1], frame.shape[0]))
                    
                    # Create side-by-side display
                    display_frame = np.hstack((frame, last_density_color))
                else:
                    count = LWCC.get_count(temp_img_path, model=model)
                    display_frame = frame.copy()
            except Exception as e:
                print(f"Inference error: {e}")
                display_frame = frame.copy()
        else:
            # For intermediate frames, re-apply the last density overlay if it exists
            if show_density and last_density_color is not None:
                display_frame = np.hstack((frame, last_density_color))
            else:
                display_frame = frame.copy()

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        
        # Overlay Info (Round the count to nearest whole number)
        cv2.putText(display_frame, f"Count: {int(round(count))}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(display_frame, f"Model: {model_name} ({model_weights})", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("LWCC Real-Time Crowd Counting", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LWCC Real-Time Counter")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (0) or video file path")
    parser.add_argument("--model", type=str, default="DM-Count", help="Model name")
    parser.add_argument("--weights", type=str, default="SHB", help="Weights (SHA for dense, SHB for sparse)")
    parser.add_argument("--density", action="store_true", help="Overlay density map heatmap")
    
    args = parser.parse_args()
    
    source = int(args.source) if args.source.isdigit() else args.source
    realtime_counting(args.model, args.weights, source, args.density)
