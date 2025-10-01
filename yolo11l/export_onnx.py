import os
import ssl
import urllib3
from ultralytics import YOLO

def main():
    try:
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, os.path.pardir)

        model_path = os.path.join(project_root, 'yolo11l', 'models', 'yolo11l.pt')
        
        # Disable SSL verification (to resolve SSL issues due to corporate firewalls, etc.)
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Set offline mode
        os.environ['YOLO_OFFLINE'] = '1'
        os.environ['ULTRALYTICS_OFFLINE'] = '1'

        # Load YOLOv11l model
        model = YOLO(model=model_path)

        # Convert model to ONNX format 
        # Using default opset (18+) to avoid version conversion issues
        # opset=12 causes compatibility issues with Resize operators in YOLOv11
        model.export(format='onnx')

        print("Successfully converted YOLOv11l model to ONNX format. 'models/yolo11l.onnx' file has been created.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
