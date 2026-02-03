import torch
import torch.nn as nn
import sys
import os

# Add paths to sys.path to enable imports
# We try to detect the current directory and add subdirectories
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'DPI'))
sys.path.append(os.path.join(current_dir, 'DPIPET'))
sys.path.append(os.path.join(current_dir, 'ml-tarflow'))

# Placeholder imports - these might need adjustment based on exact Python pathing
try:
    from DPI.DPItorch.generative_model.realnvpfc_model import RealNVP
except ImportError:
    try:
        from DPItorch.generative_model.realnvpfc_model import RealNVP
    except ImportError:
        print("Warning: Could not import RealNVP")
        RealNVP = None

try:
    from DPI.DPItorch.generative_model.glow_model import Glow
except ImportError:
    try:
        from DPItorch.generative_model.glow_model import Glow
    except ImportError:
        print("Warning: Could not import Glow")
        Glow = None

try:
    from transformer_flow import Model as TarflowModel
except ImportError:
    print("Warning: Could not import TarflowModel")
    TarflowModel = None

# MeanFlow imports
try:
    sys.path.append(os.path.join(current_dir, 'py-meanflow/meanflow'))
    from models.meanflow import MeanFlow
    from models.unet import SongUNet
    from types import SimpleNamespace
except ImportError:
    print("Warning: Could not import MeanFlow")
    MeanFlow = None
    SongUNet = None


class UnifiedFlowModel(nn.Module):
    """
    Abstract Base Class for Unified Flow Models.
    """
    def forward(self, x, condition=None, **kwargs):
        """
        Forward pass (Inference / Normalizing direction: Data -> Latent).
        Args:
            x: Input tensor.
            condition: Optional condition tensor.
            **kwargs: Additional arguments for specific models.
        Returns:
            z: Latent representation.
            logdet: Log determinant of Jacobian.
            other: Optional dictionary with other outputs (e.g., intermediate states).
        """
        raise NotImplementedError

    def inverse(self, z, condition=None, **kwargs):
        """
        Inverse pass (Generative direction: Latent -> Data).
        Args:
            z: Latent tensor.
            condition: Optional condition tensor.
            **kwargs: Additional arguments (e.g. guidance).
        Returns:
            x: Reconstructed/Generated input.
        """
        raise NotImplementedError
    
    def get_loss(self, z, logdet, **kwargs):
        """
        Calculate loss (typically Negative Log Likelihood).
        Args:
            z: Latent tensor.
            logdet: Log determinant.
        """
        # Default implementation assuming standard NLL: 0.5 * z^2 - logdet
        # Specific models might override this (e.g. Tarflow uses mean)
        nll = 0.5 * z.pow(2).sum() - logdet.sum()
        return nll / z.shape[0]

    def sample(self, num_samples, device, condition=None, input_shape=None, **kwargs):
        """
        Convenience method to sample from the prior and generate.
        """
        # This is a generic implementation, can be overridden
        if input_shape is None:
             raise ValueError("input_shape must be provided for generic sampling")
        
        z = torch.randn(num_samples, *input_shape, device=device)
        return self.inverse(z, condition, **kwargs)


class RealNVPAdapter(UnifiedFlowModel):
    def __init__(self, ndim, n_flow, affine=True, seqfrac=4, permute='random', batch_norm=True):
        super().__init__()
        if RealNVP is None:
            raise ImportError("RealNVP class not found.")
        self.model = RealNVP(
            ndim=ndim,
            n_flow=n_flow,
            affine=affine,
            seqfrac=seqfrac,
            permute=permute,
            batch_norm=batch_norm
        )
    
    def forward(self, x, condition=None, **kwargs):
        # RealNVP forward returns (out, logdet)
        z, logdet = self.model(x)
        return z, logdet, {}

    def inverse(self, z, condition=None, **kwargs):
        x, logdet = self.model.reverse(z)
        return x


class GlowAdapter(UnifiedFlowModel):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()
        if Glow is None:
             raise ImportError("Glow class not found.")
        self.model = Glow(
            in_channel=in_channel,
            n_flow=n_flow,
            n_block=n_block,
            affine=affine,
            conv_lu=conv_lu
        )

    def forward(self, x, condition=None, **kwargs):
        # Glow.forward(input) -> log_p_sum, logdet, z_outs
        # z_outs is a list of tensors [z1, z2, ..., z_final]
        log_p_sum, logdet, z_outs = self.model(x)
        return z_outs, logdet, {'log_p_sum': log_p_sum}

    def inverse(self, z, condition=None, **kwargs):
        # Glow.reverse(z_list) -> input, logdet
        if not isinstance(z, list):
            raise TypeError("GlowAdapter.inverse expects a list of tensors for z")
        x, logdet = self.model.reverse(z)
        return x
    
    def get_loss(self, z, logdet, **kwargs):
        # Glow z is a list of tensors
        if isinstance(z, list):
            nll = 0
            for z_i in z:
                nll += 0.5 * z_i.pow(2).sum()
            nll = nll - logdet.sum()
            # Normalize by batch size
            return nll / z[0].shape[0]
        else:
            return super().get_loss(z, logdet, **kwargs)


class TarflowAdapter(UnifiedFlowModel):
    def __init__(self, in_channels, img_size, patch_size, channels, num_blocks, layers_per_block, nvp=True, num_classes=0):
        super().__init__()
        if TarflowModel is None:
             raise ImportError("TarflowModel class not found.")
        self.model = TarflowModel(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            channels=channels,
            num_blocks=num_blocks,
            layers_per_block=layers_per_block,
            nvp=nvp,
            num_classes=num_classes
        )

    def forward(self, x, condition=None, **kwargs):
        # Tarflow forward(x, y) -> x, outputs, logdets
        z, outputs, logdets = self.model(x, y=condition)
        return z, logdets, {'intermediate_outputs': outputs}

    def inverse(self, z, condition=None, **kwargs):
        # Tarflow reverse accepts kwargs like guidance
        x_recon = self.model.reverse(z, y=condition, **kwargs)
        return x_recon

    def get_loss(self, z, logdet, **kwargs):
        return self.model.get_loss(z, logdet)


class MeanFlowAdapter(UnifiedFlowModel):
    def __init__(self, img_resolution=32, in_channels=3, out_channels=3,
                 dropout=0.2, ratio=0.75, ema_decay=0.9999, 
                 ema_decays=[0.99995, 0.9996], norm_p=0.75, norm_eps=1e-3,
                 channel_mult=[2, 2, 2], **kwargs):
        super().__init__()
        if MeanFlow is None or SongUNet is None:
            raise ImportError("MeanFlow or SongUNet class not found.")
        
        # Create args namespace with required parameters
        args = SimpleNamespace(
            ratio=ratio,
            dropout=dropout,
            ema_decay=ema_decay,
            ema_decays=ema_decays,
            norm_p=norm_p,
            norm_eps=norm_eps,
            use_edm_aug=False,
            tr_sampler='v1',
            P_mean_t=-0.6,
            P_std_t=1.6,
            P_mean_r=-4.0,
            P_std_r=1.6
        )
        
        # Network configuration
        net_configs = {
            'img_resolution': img_resolution,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'channel_mult_noise': 2,
            'resample_filter': [1, 3, 3, 1],
            'channel_mult': channel_mult,
            'encoder_type': 'standard',
            'decoder_type': 'standard',
            'dropout': dropout,
        }
        
        self.model = MeanFlow(arch=SongUNet, args=args, net_configs=net_configs)
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x, condition=None, **kwargs):
        """
        Forward pass with loss computation.
        Returns:
            z: Not applicable for MeanFlow (returns None)
            logdet: Not applicable for MeanFlow (returns None)  
            other: Dictionary containing 'loss'
        """
        # MeanFlow's forward_with_loss expects (x, aug_cond)
        aug_cond = condition
        loss = self.model.forward_with_loss(x, aug_cond)
        return None, None, {'loss': loss}
    
    def inverse(self, z=None, condition=None, **kwargs):
        """
        Sample from the model.
        For MeanFlow, z is ignored and sampling is done from scratch.
        """
        device = kwargs.get('device', 'cpu')
        num_samples = kwargs.get('num_samples', 1)
        
        samples_shape = (num_samples, self.in_channels, 
                        self.img_resolution, self.img_resolution)
        
        return self.model.sample(samples_shape, device=device)
    
    def get_loss(self, z, logdet, **kwargs):
        """
        Get loss from forward pass results.
        For MeanFlow, the loss is stored in kwargs.
        """
        if 'loss' in kwargs:
            return kwargs['loss']
        # If not in kwargs, need to compute it
        return 0.0
    
    def sample(self, num_samples, device, condition=None, input_shape=None, **kwargs):
        """
        Convenience method to sample from the prior and generate.
        """
        if input_shape is None:
            input_shape = (self.in_channels, self.img_resolution, self.img_resolution)
        
        samples_shape = (num_samples,) + input_shape
        return self.model.sample(samples_shape, device=device)


class ModelFactory:
    @staticmethod
    def create_model(model_type, config):
        """
        Factory method to create a unified model.
        Args:
            model_type: str, one of ['realnvp', 'glow', 'tarflow']
            config: dict, configuration parameters for the specific model
        Returns:
            UnifiedFlowModel instance
        """
        model_type = model_type.lower()
        if model_type == 'realnvp':
            return RealNVPAdapter(**config)
        elif model_type == 'glow':
            return GlowAdapter(**config)
        elif model_type == 'tarflow':
            return TarflowAdapter(**config)
        elif model_type == 'meanflow':
            return MeanFlowAdapter(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Example usage:
if __name__ == "__main__":
    # Example Configs
    # Note: ndim must be large enough relative to seqfrac to avoid 0 dimension in hidden layers
    realnvp_config = {
        'ndim': 32,
        'n_flow': 4,
        'affine': True,
        'seqfrac': 4
    }
    
    try:
        model = ModelFactory.create_model('realnvp', realnvp_config)
        print("RealNVP Model created successfully")
        # Dummy forward
        x = torch.randn(10, 32)
        z, logdet, _ = model.forward(x)
        print(f"Forward pass successful. z shape: {z.shape}, logdet shape: {logdet.shape}")
        x_recon = model.inverse(z)
        print(f"Inverse pass successful. x_recon shape: {x_recon.shape}")
    except Exception as e:
        print(f"RealNVP Test Failed: {e}")
        import traceback
        traceback.print_exc()
