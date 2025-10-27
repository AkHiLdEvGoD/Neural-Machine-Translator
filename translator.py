from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
import onnxruntime as ort
import json

class ONNXTranslator:
    def __init__(self,model_dir):
        model_dir = Path(model_dir)
        print("Loading ONNX models...")

        with open(model_dir/"config.json",'r') as f:
            self.config = json.load(f)

        self.sos_id = self.config['sos_id']
        self.eos_id = self.config['eos_id']
        self.pad_id = self.config['pad_id']

        self.tokenizer = Tokenizer.from_file(str(model_dir/"tokenizer.json"))
        self.encoder_session = ort.InferenceSession(
            str(model_dir / "encoder.onnx"),
            providers=['CPUExecutionProvider']  
        )
        self.decoder_session = ort.InferenceSession(
            str(model_dir / "decoder.onnx"),
            providers=['CPUExecutionProvider']  # CPU only
        )
        
        print("ONNX models loaded successfully!")

ONNXTranslator('assets/ONNX_models')