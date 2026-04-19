import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExporter:
    
    @staticmethod
    def export_to_onnx(model, model_path: str, output_path: str, input_shape: tuple = (1, 10, 99)):
        device = next(model.parameters()).device
        model.eval()
        
        dummy_input = torch.randn(input_shape, device=device)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}},
            verbose=False
        )
        
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info(f"ONNX model exported to {output_path}")

    @staticmethod
    def export_to_torchscript(model, output_path: str):
        model.eval()
        scripted_model = torch.jit.script(model)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        scripted_model.save(output_path)
        logger.info(f"TorchScript model exported to {output_path}")

    @staticmethod
    def export_to_quantized(model, output_path: str, device: str = "cpu"):
        model.eval()
        model = model.to(device)
        
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec={torch.nn.LSTM},
            dtype=torch.qint8
        )
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized_model.state_dict(), output_path)
        logger.info(f"Quantized model saved to {output_path}")

    @staticmethod
    def export_model_metadata(model_config: dict, metadata_path: str):
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        logger.info(f"Model metadata saved to {metadata_path}")

    @staticmethod
    def verify_onnx_model(onnx_path: str, test_input: np.ndarray):
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        result = session.run([output_name], {input_name: test_input})
        logger.info(f"ONNX model inference successful: {result[0].shape}")
        return result[0]

    @staticmethod
    def export_ensemble(temporal_model, static_model, output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        ensemble_config = {
            'temporal_model': 'temporal_model.pt',
            'static_model': 'static_model.pt',
            'fusion_method': 'average'
        }
        
        torch.save(temporal_model.state_dict(), Path(output_dir) / 'temporal_model.pt')
        torch.save(static_model.state_dict(), Path(output_dir) / 'static_model.pt')
        
        with open(Path(output_dir) / 'ensemble_config.json', 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        logger.info(f"Ensemble models exported to {output_dir}")
