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

    def create_padding_mask(self,seq):
        mask = (seq != self.pad_id).astype(np.bool_)
        mask = np.expand_dims(mask,axis=(1,2))
        return mask
        
    def create_causal_mask(self,seq_len):
        mask = np.tril(np.ones((seq_len,seq_len),dtype=np.bool_))
        mask = np.expand_dims(mask,axis=(0,1))
        return mask
        
    def translate(self,sentence,max_len=100):
        src_tokens = self.tokenizer.encode(sentence).ids
        src = np.array([[self.sos_id]+src_tokens+[self.eos_id]],dtype=np.int64)
        src_mask = self.create_padding_mask(src)

        enc_output = self.encoder_session.run(
            ['enc_output'],
            {'src':src,'src_mask':src_mask}
        )[0]

        tgt_token = [self.sos_id]

        for _ in range(max_len):
            tgt = np.array([tgt_token],dtype=np.int64)
            tgt_padding_mask = self.create_padding_mask(tgt)
            tgt_causal_mask = self.create_causal_mask(len(tgt_token))
            tgt_mask = tgt_padding_mask & tgt_causal_mask

            logits = self.decoder_session.run(
                ['logits'],
                {
                    'tgt':tgt,
                    'enc_output':enc_output,
                    'tgt_mask':tgt_mask,
                    'src_mask':src_mask
                }
            )[0]

            next_token = np.argmax(logits[0,-1,:])
            tgt_token.append(int(next_token))

            if next_token == self.eos_id:
                break
        
        output_token = [t for t in tgt_token[1:] if t not in [self.sos_id,self.eos_id,self.pad_id]]
        translation =  self.tokenizer.decode(output_token)

        return translation
    