import torch
import torch.nn as nn


# Standard autoencoder configurations
AUTOENCODER_CONFIGS = {
	'cifar32': {
		'input_channels': 3,
		'input_size': 32,
		'encoder_channels': [32, 64, 128],
		'encoder_kernels': [3, 3, 3],
		'encoder_strides': [2, 2, 2],
		'encoder_paddings': [1, 1, 1],
		'bottleneck_spatial': 4,
		'latent_dim': 64,
		'decoder_channels': [64, 32, 3],
		'decoder_kernels': [3, 3, 3],
		'decoder_strides': [2, 2, 2],
		'decoder_paddings': [1, 1, 1],
		'decoder_output_paddings': [1, 1, 1],
		'use_sigmoid': True,
		'description': 'CIFAR-style 32x32 RGB images with 64-dim latent space'
	},
	"mnist32": {
		"input_channels": 3,
		"input_size": 32,
		"encoder_channels": [32, 64],
		"encoder_kernels": [3, 3],
		"encoder_strides": [2, 2],
		"encoder_paddings": [1, 1],
		"bottleneck_spatial": 8,
		"latent_dim": 32,
		"decoder_channels": [32, 3],
		"decoder_kernels": [3, 3],
		"decoder_strides": [2, 2],
		"decoder_paddings": [1, 1],
		"decoder_output_paddings": [1, 1],
		"use_sigmoid": True,
		"description": "MNIST-style 32x32 RGB images with 32-dim latent space"
	}
}


class Encoder(nn.Module):
	"""
	Generic encoder network for autoencoder.
	Builds convolutional layers based on configuration.
	"""
	
	def __init__(self, config):
		super(Encoder, self).__init__()
		self.config = config
		self.latent_dim = config['latent_dim']
		
		# Build convolutional layers
		layers = []
		in_channels = config['input_channels']
		
		for i, out_channels in enumerate(config['encoder_channels']):
			layers.append(nn.Conv2d(
				in_channels, 
				out_channels,
				kernel_size=config['encoder_kernels'][i],
				stride=config['encoder_strides'][i],
				padding=config['encoder_paddings'][i]
			))
			layers.append(nn.ReLU())
			in_channels = out_channels
		
		self.conv_layers = nn.Sequential(*layers)
		
		# Calculate flattened size
		final_channels = config['encoder_channels'][-1]
		spatial_size = config['bottleneck_spatial']
		flattened_size = final_channels * spatial_size * spatial_size
		
		# Linear layer to latent space
		self.fc = nn.Linear(flattened_size, config['latent_dim'])
	
	def forward(self, x):
		x = self.conv_layers(x)
		x = x.view(x.size(0), -1)  # Flatten
		x = self.fc(x)
		return x


class Decoder(nn.Module):
	"""
	Generic decoder network for autoencoder.
	Builds transposed convolutional layers based on configuration.
	"""
	
	def __init__(self, config):
		super(Decoder, self).__init__()
		self.config = config
		self.latent_dim = config['latent_dim']
		self.use_sigmoid = config.get('use_sigmoid', True)
		
		# Calculate the initial spatial dimensions
		initial_channels = config['encoder_channels'][-1]
		spatial_size = config['bottleneck_spatial']
		flattened_size = initial_channels * spatial_size * spatial_size
		
		# Linear layer from latent space
		self.fc = nn.Linear(config['latent_dim'], flattened_size)
		self.initial_channels = initial_channels
		self.spatial_size = spatial_size
		
		# Build transposed convolutional layers
		layers = []
		in_channels = initial_channels
		
		for i, out_channels in enumerate(config['decoder_channels']):
			layers.append(nn.ConvTranspose2d(
				in_channels,
				out_channels,
				kernel_size=config['decoder_kernels'][i],
				stride=config['decoder_strides'][i],
				padding=config['decoder_paddings'][i],
				output_padding=config['decoder_output_paddings'][i]
			))
			
			# Add ReLU for all but the last layer
			if i < len(config['decoder_channels']) - 1:
				layers.append(nn.ReLU())
			
			in_channels = out_channels
		
		self.deconv_layers = nn.Sequential(*layers)
		self.activation = nn.Sigmoid() if self.use_sigmoid else nn.Identity()
	
	def forward(self, x):
		x = self.fc(x)
		x = x.view(x.size(0), self.initial_channels, self.spatial_size, self.spatial_size)
		x = self.deconv_layers(x)
		x = self.activation(x)
		return x


class Autoencoder(nn.Module):
	"""
	Complete Autoencoder combining Encoder and Decoder.
	"""
	
	def __init__(self, config):
		super(Autoencoder, self).__init__()
		self.config = config
		self.latent_dim = config['latent_dim']
		self.encoder = Encoder(config)
		self.decoder = Decoder(config)
	
	def forward(self, x):
		latent = self.encoder(x)
		reconstructed = self.decoder(latent)
		return reconstructed
	
	def encode(self, x):
		"""Get latent representation."""
		return self.encoder(x)
	
	def decode(self, z):
		"""Reconstruct from latent representation."""
		return self.decoder(z)


def create_autoencoder(config_name='cifar32', config=None, device='cpu', pretrained_path=None):
	"""
	Helper function to construct and load the autoencoder model onto a desired device.
	"""

	device = torch.device(device)
	
	if config is not None:
		model_config = config
		print(f"Using custom configuration")
	elif config_name in AUTOENCODER_CONFIGS:
		model_config = AUTOENCODER_CONFIGS[config_name]
		print(f"Using configuration '{config_name}': {model_config.get('description', 'N/A')}")
	else:
		available_configs = ', '.join(AUTOENCODER_CONFIGS.keys())
		raise ValueError(f"Unknown config_name '{config_name}'. Available: {available_configs}")
	
	model = Autoencoder(model_config)
	
	if pretrained_path is not None:
		checkpoint = torch.load(pretrained_path, map_location=device)
		
		if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
			model.load_state_dict(checkpoint['model_state_dict'])
		elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
			model.load_state_dict(checkpoint['state_dict'])
		else:
			model.load_state_dict(checkpoint)
		
		print(f"Loaded pretrained model from {pretrained_path}")
	
	model = model.to(device)
	return model
