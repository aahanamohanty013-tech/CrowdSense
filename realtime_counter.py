import cv2
import argparse
import time
import numpy as np
from lwcc import LWCC

def realtime_counting(model_name="DM-Count", model_weights="SHA", source=0, show_density=False):
    """
    Captures video from webcam or file and performs real-time crowd counting.
    """
    print(f"Loading model {model_name}...")
    model = LWCC.load_model(model_name=model_name, model_weights=model_weights)
    
    # Initialize capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print("Starting real-time capture. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            from PIL import Image
            pil_img = Image.fromarray(rgb_frame)
            
            if show_density:
                count, density_map = LWCC.get_count(pil_img, model=model, return_density=True)
                
                # Normalize density map for visualization
                density_norm = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
                density_color = cv2.applyColorMap(density_norm.astype(np.uint8), cv2.COLORMAP_JET)
                
                # Resize density map to match frame size
                density_color = cv2.resize(density_color, (frame.shape[1], frame.shape[0]))
                
                # Blend with original frame
                display_frame = cv2.addWeighted(frame, 0.6, density_color, 0.4, 0)
            else:
                count = LWCC.get_count(pil_img, model=model)
                display_frame = frame
                
        except Exception as e:
            print(f"Inference error: {e}")
            count = 0
            display_frame = frame

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        
        # Overlay Info
        cv2.putText(display_frame, f"Count: {count:.2f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.imshow("LWCC Real-Time Crowd Counting", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LWCC Real-Time Counter")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (0) or video file path")
    parser.add_argument("--model", type=str, default="DM-Count", help="Model name")
    parser.add_argument("--weights", type=str, default="SHA", help="Weights (SHA/SHB)")
    parser.add_argument("--density", action="store_true", help="Overlay density map heatmap")
    
    args = parser.parse_args()
    
    source = int(args.source) if args.source.isdigit() else args.source
    realtime_counting(args.model, args.weights, source, args.density)
