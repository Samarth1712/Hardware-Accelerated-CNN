import os
import argparse
import torch
import cv2
import numpy as np
from pytorch_nndct.apis import torch_quantizer
from models.experimental import attempt_load

def preprocess_image(image_path, input_size=(640, 640)):
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.resize(img, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0)
    return img

def quantize(args):
    device = torch.device("cpu")
    print(f"Loading model: {args.model}")
    try:
        model = attempt_load(args.model, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    # Create a dummy input for the forward pass
    input_tensor = torch.randn([1, 3, 640, 640], dtype=torch.float32).to(device)

    quantizer = torch_quantizer(
        args.mode, model, (input_tensor), 
        output_dir=args.output_dir, 
        target="DPUCZDX8G_ISA1_B4096"
    )

    quantized_model = quantizer.quant_model

    if args.mode == 'calib':
        print("Starting calibration...")
        if not os.path.exists(args.image_dir):
            print(f"Image dir {args.image_dir} not found!")
            return
            
        image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.jpg', '.png'))][:200]
        count = 0
        with torch.no_grad():
            for f in image_files:
                img = preprocess_image(os.path.join(args.image_dir, f))
                if img is None: continue
                quantized_model(img)
                count += 1
                if count % 20 == 0: print(f"Processed {count}...")
        
        quantizer.export_quant_config()
        print("Calibration done.")

    elif args.mode == 'test':
        print("Exporting xmodel...")
        # CRITICAL FIX: Run a forward pass so Vitis knows the graph structure
        with torch.no_grad():
            quantized_model(input_tensor)
            
        quantizer.export_xmodel(deploy_check=False, output_dir=args.output_dir)
        print("Export done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best.pt')
    parser.add_argument('--image_dir', type=str, default='images')
    parser.add_argument('--output_dir', type=str, default='quantized_result')
    parser.add_argument('--mode', type=str, default='calib')
    args = parser.parse_args()
    quantize(args)
