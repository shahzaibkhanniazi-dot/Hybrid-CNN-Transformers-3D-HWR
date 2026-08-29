import torch
import torch.nn.functional as F
import numpy as np

class GradCAM1D:
    """
    1D Temporal Grad-CAM for IMU Kinematic Signals.
    Extracts gradient activations across temporal convolutional layers.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, signal, target_class_idx=None):
        """
        signal: [1, Channels, Time]
        Returns: 1D heatmap array normalized between [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()
        
        outputs = self.model(signal)
        logits = outputs["ctc_logits"]  # [1, T', Vocab]
        
        if target_class_idx is None:
            target_class_idx = logits.argmax(dim=-1).mean().long()
            
        score = logits[:, :, target_class_idx].sum()
        score.backward(retain_graph=True)
        
        # Global Average Pooling of gradients along time axis: [1, C, 1]
        pooled_grads = torch.mean(self.gradients, dim=2, keepdim=True)
        
        # Weight activations by pooled gradients: [1, C, T']
        cam = torch.sum(pooled_grads * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU
        
        # Upsample back to original input time length
        cam = F.interpolate(cam, size=signal.shape[2], mode='linear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize between 0 and 1
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam