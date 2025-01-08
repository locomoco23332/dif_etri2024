import os
# from turtle import forward 
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
# print(parent_dir)
# print(current_dir)
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
#import pykan.kan.KANLayer as kn
#import KANU_Net.src.kan_unet as ku
#import kan
#import matplotlib.pyplot as plt
#import numpy as np
#import torch
#from kan import KAN, create_dataset
#from sklearn.datasets import make_classification, load_iris
#from sklearn.model_selection import train_test_split
def Tonumpy(data):
    if data.ref().ndim == 1 :
        return np.array(data.ref())
    else :
        numpy_data = np.empty((data.rows(),data.cols()))
        numpy_data[:data.rows(),:data.cols()]=data.ref()
        return numpy_data
class NormalNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.fckan= kn.KANLayer(in_dim=input_size,out_dim=output_size)
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        self.fc4 = nn.Linear(hidden_size,output_size)
#        self.fc5 = nn.Linear(hidden_size,hidden_size)
#        self.fc6 = nn.Linear(hidden_size,output_size)

    def forward(self,input_data):
        input_data = input_data.flatten(-2)
        out = self.fc1(F.elu(input_data))
        out = self.fc2(F.elu(out))
        out = self.fc3(F.elu(out))
#        out4 = self.fc4(F.elu(out3))
#        out5 = self.fc4(F.elu(out4))
        return self.fc4(out)

    def set_normalization(self,mu,std):
        self.mu = mu
        self.std = std

    def normalize(self,t):
        return (t-self.mu)/self.std

    def denormalize(self,t):
        return t*self.std + self.mu

class Encoder2(nn.Module):
    def __init__(self,input_size,latent_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = 256
        self.num_layers = 1

        #self.device="cuda"
        real_input = self.input_size+35
        #self.gru1 = nn.GRU(real_input,self.hidden_size,batch_first=True)
        #self.gru(self.hidden_size+self.input_size,
        self.fc1 = nn.Linear(real_input,self.hidden_size)
        self.fc2 = nn.Linear(input_size+self.hidden_size,self.hidden_size)
        self.mu  = nn.Linear(input_size+self.hidden_size,latent_size)
        self.var = nn.Linear(input_size+self.hidden_size,latent_size)
    
    def reparameterize(self,mu,var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self,input,condition_input):
        out1 = F.elu(self.fc1(torch.cat((input,condition_input),dim=1)))
        out2 = F.elu(self.fc2(torch.cat((input,out1),dim=1)))
        out3 = torch.cat((input,out2),dim=1)
        return self.mu(out3) , self.var(out3)

    def forward(self,input, condition_input):
        mu , var = self.encode(input,condition_input)
        z = self.reparameterize(mu,var)
        return z,mu,var
class Encoder3(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = 256
        self.num_layers = 1
        
        # Input includes both the input size and the condition input size (+35)
        real_input = self.input_size + 35
        
        # Define GRU layer to process the concatenated inputs
        self.gru = nn.GRU(real_input, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
        # Linear layers to generate mean and variance from GRU outputs
        self.mu = nn.Linear(self.hidden_size, latent_size)
        self.var = nn.Linear(self.hidden_size, latent_size)
    
    def reparameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, input, condition_input):
        # Concatenate input and condition_input
        combined_input = torch.cat((input, condition_input), dim=1)  # dim=2 for GRU (batch, seq_len, features)
        
        # Pass through GRU
        gru_output, _ = self.gru(combined_input)  # Output shape: (batch_size, seq_len, hidden_size)
        
        # Use the output of the last time step
        out_last = gru_output[:, -1]  # Shape: (batch_size, hidden_size)
        print(out_last.shape)
        # Compute mean and variance
        mu = self.mu(out_last)  # Shape: (batch_size, latent_size)
        var = self.var(out_last)  # Shape: (batch_size, latent_size)
        
        return mu, var
    
    def forward(self, input, condition_input):
        # Encode the input and condition input
        mu, var = self.encode(input, condition_input)
        
        # Reparameterize to get latent variable z
        z = self.reparameterize(mu, var)
        
        return z, mu, var


class Encoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = 256
        self.num_layers = 1

        real_input = self.input_size + 35
        
        # GRU layer
        self.gru = nn.GRU(input_size=real_input, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
        # Fully connected layers
        self.fc2 = nn.Linear(latent_size + self.hidden_size, self.hidden_size)
        self.mu = nn.Linear(latent_size + self.hidden_size, 35)
        self.var = nn.Linear(latent_size + self.hidden_size, 35)

    def reparameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, input, condition_input):
        # Concatenate input and condition_input along the last dimension
        #print(input.shape)
        gru_input = torch.cat((input, condition_input), dim=-1)
        
        # Pass through GRU
        gru_output, _ = self.gru(gru_input)
        
        # Take the last hidden state output
        #print(gru_output.shape)
        
        # Pass through fully connected layers
        out2 = F.elu(self.fc2(torch.cat(( gru_output,condition_input), dim=-1)))
        out3 = torch.cat((out2,condition_input), dim=1)
        
        return self.mu(out3), self.var(out3)

    def forward(self, input, condition_input):
        mu, var = self.encode(input, condition_input)
        z = self.reparameterize(mu, var)
        return z, mu, var

class Decoder(nn.Module):
    def __init__(self,input_size,latent_size,num_experts,output_size):
        super().__init__()
        self.input_size = input_size+latent_size
        output_size = output_size
        hidden_size = 256
        self.fc1 = nn.Linear(self.input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size+latent_size,hidden_size)
        self.fc3 = nn.Linear(latent_size+hidden_size,output_size)

    def forward(self,z,condition_input):
        # print("latent_shape : ",z.shape)
        out4 = F.elu(self.fc1(torch.cat((z,condition_input),dim=1)))
        out5 = F.elu(self.fc2(torch.cat((z,out4),dim=1)))
        return self.fc3(torch.cat((z,out5),dim=1))

class Decoder2(nn.Module):
    def __init__(self,input_size,latent_size,num_experts,output_size):
        super().__init__()
        self.input_size = input_size+latent_size
        output_size = output_size
        hidden_size = 256
        self.fc1 = nn.Linear(self.input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size+input_size,hidden_size)
        self.fc3 = nn.Linear(input_size+hidden_size,output_size)

    def forward(self,z,condition_input):
        # print("latent_shape : ",z.shape)
        out4 = F.elu(self.fc1(torch.cat((z,condition_input),dim=1)))
        out5 = F.elu(self.fc2(torch.cat((z,out4),dim=1)))
        return self.fc3(torch.cat((z,out5),dim=1))
class MixedDecoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + frame_size * num_condition_frames
        inter_size = latent_size + hidden_size
        output_size = num_future_predictions * frame_size
        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z ):
        coefficients = F.softmax(self.gate(z), dim=1)
        for (weight, bias, activation) in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = z.unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out



class VAE(nn.Module):
    def __init__(self,input_size,latent_size,num_experts,output_size):
        super().__init__()
        self.encoder = Encoder(input_size,latent_size)
        #self.decoder = MixedDecoder(input_size,latent_size,256,1,1,num_experts)
        self.decoder = Decoder(input_size,latent_size,num_experts,output_size)
        
        ############################change initialization orer##########################3
        self.data_std = 0
        self.data_avg = 0
        ############################change initialization orer##########################3
        self.latent_list = []

    def encode(self,x,c):
        z,mu,logvar = self.encoder(x,c)
        return z,mu,logvar
    def forward(self,x,c):
        z,mu,logvar = self.encoder(x,c)
        return self.decoder(z,c),mu,logvar
    def sample(self,z,c):
        return self.decoder(z,c)
    def set_normalization(self,std,avg):
        self.data_std=std
        self.data_avg=avg
    def set_latent_list(self,latent_vectors):
        self.latent_list = latent_vectors

    #######################
    def normalize(self, t):
        return (t - self.data_avg) / self.data_std
    def denormalize(self, t):
        return t * self.data_std + self.data_avg
    #######################

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Encoder4(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder4, self).__init__()
        self.fc = nn.Linear(input_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class Decoder4(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder4, self).__init__()
        self.fc = nn.Linear(35, output_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        return self.relu(self.fc(z))

class DeFeeNetWithEncoderDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super(DeFeeNetWithEncoderDecoder, self).__init__()
        self.encoder = Encoder(input_dim, 35)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc =nn.Linear(hidden_dim+input_dim,hidden_dim)
        self.gelu = nn.GELU()
        self.decoder = Decoder(35,35,num_experts=6,output_size=output_dim)
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_var_layer = nn.Linear(hidden_dim, output_dim)
        self.data_std = 0
        self.data_avg = 0


    def forward(self, obs, prev_pred, prev_obs,future):
        """
        :param obs: 현재 관찰 \(X_r\) [batch_size, N, input_dim]
        :param prev_pred: 이전 예측 [batch_size, T, input_dim]
        :param prev_obs: 이전 관찰 [batch_size, T, input_dim]
        """
        # 오차 계산
        velocity_obs = prev_obs- prev_pred
        velocity_pred = future - prev_pred
        deviation = velocity_obs - velocity_pred
        
        # 현재 관찰 데이터를 인코딩
        encoded_obs,mean,var = self.encoder(obs,future)
        
        # GRU로 오차 학습
        gru_out, _ = self.gru(deviation)
        
        # 잠재 공간 결합
        combined_latent = torch.cat((encoded_obs, gru_out), dim=-1)
         # 디코딩하여 최종 예측 생성
        hidden_state = self.fc(combined_latent)
        hidden_state = self.gelu(hidden_state)

        # Get mean and log variance for Gaussian distribution
        mean = self.mean_layer(hidden_state)
        log_var = self.log_var_layer(hidden_state)
        std = torch.exp(0.5 * log_var)
        
        
        # Sample using reparameterization trick
        eps = torch.randn_like(std)
        sampled_output = mean + eps * std
        
        # Decode the sampled output
        output = self.decoder(sampled_output,future)
        
        return output,mean,log_var
    def encode(self,x,c):
        z,mu,logvar = self.encoder(x,c)
        return z,mu,logvar
    def sample(self,z,c):
        return self.decoder(z,c)
    def set_normalization(self,std,avg):
        self.data_std=std
        self.data_avg=avg
    def set_latent_list(self,latent_vectors):
        self.latent_list = latent_vectors
     #######################
    def normalize(self, t):
        return (t - self.data_avg) / self.data_std
    def denormalize(self, t):
        return t * self.data_std + self.data_avg
    #######################
class BetaDerivatives():
    def __init__(self,time_steps,beta_start,beta_end):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.time_steps = time_steps
        
        self.betas = self.prepare_noise_schedule().to(device="cpu")
        #self.betas = self.prepare_noise_schedule().to(device="cuda")
        self.alpha = 1-self.betas
        self.alpha_hat = torch.cumprod(self.alpha,dim=0)
        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start,self.beta_end,self.time_steps)

    def sample_timesteps(self,n):
        return torch.randint(low=1,high=self.time_steps-1,size=(n,))

    def gather(self,a,t):
        return torch.gather(a,1,t)


class GaussianDiffusion():
    def __init__(self,input_size,noise_step,output_size):
        self.device = "cpu"
        #self.device = "cuda"
        self.input_size = input_size
        self.output_size = output_size
        self.noise_step = noise_step
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.betaderivative = BetaDerivatives(noise_step,self.beta_start,self.beta_end)
        
        self.beta = self.betaderivative.prepare_noise_schedule().to(self.device)
        self.alpha = self.betaderivative.alpha
        self.alpha_hat = self.betaderivative.alpha_hat

    def q_sample(self,x_0,t,noise=None):
        if noise is None:
            noise = torch.randn((t.shape[0],x_0.shape[0]))
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alpha_hat[t])
        return sqrt_alpha_hat*x_0 + sqrt_one_minus_alpha_hat*noise,noise.to(self.device)

class TimeEmbedding(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.n = n
        self.dim= n//2
        self.fc1 = nn.Linear(n,n)
        self.fc2 = nn.Linear(n,self.dim)

    def activation(self,x):
        return x*F.elu(x)

    def forward(self,t):
        half_dim = self.n//2
        emb = torch.log(torch.tensor(10000.0,device="cpu")/(half_dim-1)).to(device="cpu")
        emb = torch.exp(torch.arange(half_dim,device="cpu")*-emb).to(device="cpu")
        emb = t*emb
        emb = torch.cat((emb.sin(),emb.cos()),dim=1)
        emb = self.activation(self.fc1(emb))
        emb = self.fc2(emb)
        return emb

import torch
from torch import nn
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-2)
        v = v / v.size(-2)

        context = torch.einsum('b h d j, b h d i -> b h i j', v, k)
        out = torch.einsum('b h i j, b h d j -> b h d i', context, q)

        # Since 'i' = 1, we can remove that dimension by squeezing it
        out = out.squeeze(-1)  # Remove the dimension where 'i' = 1

        # Now we can safely rearrange to (h d, 1, 1) for spatial dimensions
        out = rearrange(out, "b h d -> b (h d) 1 1")

        return self.to_out(out)
# Denoising Diffusion Model using UNet and GaussianDiffusion



class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
class DownsamplingBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=4, num_encoder_layers=2, num_decoder_layers=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

        # Transformer component
        self.transformer = nn.Transformer(
            d_model=out_channels,  # Matches the out_channels of conv1d
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )

    def forward(self, x):
        # Downsampling part
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        # Preparing the input for the transformer
        # Transformer expects (sequence length, batch size, embedding dimension)
        # Permuting the tensor to have shape (sequence_length, batch_size, embedding_dim)
        x = x.permute(2, 0, 1)

        # Creating a dummy target sequence for the decoder part of the transformer
        target = torch.zeros_like(x)

        # Transformer forward pass
        x = self.transformer(x, target)

        # Permuting back to the original shape (batch_size, out_channels, sequence_length)
        x = x.permute(1, 2, 0)

        return x
# Upsampling block
class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class UpsamplingBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=4, num_encoder_layers=2, num_decoder_layers=2):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

        # Transformer component
        self.transformer = nn.Transformer(
            d_model=out_channels,  # Matches the out_channels of conv1d
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )

    def forward(self, x):
        # Upsampling part
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        # Preparing the input for the transformer
        # Transformer expects (sequence length, batch size, embedding dimension)
        # Permuting the tensor to have shape (sequence_length, batch_size, embedding_dim)
        x = x.permute(2, 0, 1)

        # Creating a dummy target sequence for the decoder part of the transformer
        target = torch.zeros_like(x)

        # Transformer forward pass
        x = self.transformer(x, target)

        # Permuting back to the original shape (batch_size, out_channels, sequence_length)
        x = x.permute(1, 2, 0)

        return x
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.attn = Attention(dim=35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.up3 = UpsamplingBlock(128,128)
        self.res7 = ResNetBlock(128,128)
        self.attn5 = Attention(dim=128)
        self.nn6= nn.Linear(64,out_channels)
        
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        xx = x
        x=self.nn1(x)

        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)


        #print(x.shape)
        x=x+t

        x=x.unsqueeze(-1)
        x=self.up0(x)
        x=self.res0(x)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2 = u2.unsqueeze(-1)
        u2= self.up3(u2)
        u2= self.res7(u2)
        u2 = u2.unsqueeze(-1)
        u2= self.attn5(u2)
        u2=u2.squeeze(-1)
        u2=u2.squeeze(-1)

        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)

        # Final output
        u3=self.final_conv(u2).squeeze(-1)
        #print(u3.shape)
        return u3
class UNetTransformer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.attn = Attention(dim=35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        #self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(99, 99)
        self.res6 = ResNetBlock(99,99)
        self.attn4 =Attention(dim=99)
        self.up3 = UpsamplingBlock(198,198)
        self.res7 = ResNetBlock(198,198)
        self.attn5 = Attention(dim=198)
        self.nn6= nn.Linear(198,out_channels)
        self.transformer = TransformerModel(in_channels=35)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t,p):
        # Downsampling
        xx = x
        x_trans=self.transformer(x,p)
        x=self.nn1(x)

        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)


        #print(x.shape)
        x=x+t

        x=x.unsqueeze(-1)
        x=self.up0(x)
        x=self.res0(x)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1= torch.cat((u1,x_trans),dim=-1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2 = torch.cat((u2,x_trans),dim=-1)
        u2 = u2.unsqueeze(-1)
        u2= self.up3(u2)
        u2= self.res7(u2)
        u2 = u2.unsqueeze(-1)
        u2= self.attn5(u2)
        u2=u2.squeeze(-1)
        u2=u2.squeeze(-1)

        #u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)

        # Final output
        u3=self.final_conv(u2).squeeze(-1)
        #print(u3.shape)
        return u3


class PUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.mid0 = nn.Linear(70,35)
        self.midd = nn.Linear(35,70)
        self.attn = Attention(dim=35)
        self.up0 = UpsamplingBlock(70,70)
        self.res0 = ResNetBlock(70,70)
        self.nn4 = nn.Linear(70,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.up3 = UpsamplingBlock(128,128)
        self.res7 = ResNetBlock(128,128)
        self.attn5 = Attention(dim=128)
        self.upd1= UpsamplingBlock(64,64)
        self.updres1= ResNetBlock(64,64)
        self.nupd1 = nn.Linear(64,out_channels)
        self.upd2 = UpsamplingBlock(128,128)
        self.updres2 =ResNetBlock(128,128)
        self.ud2n = nn.Linear(128,64)
        self.updd2 =UpsamplingBlock(64,64)
        self. upddres2 =ResNetBlock(64,64)
        self.nupd2= nn.Linear(128,64)
        self.nupdd2 = nn.Linear(64,out_channels)
        self.nn6= nn.Linear(64,out_channels)
        self.nn7 = nn.Linear(3*out_channels,out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        xx = x
        x=self.nn1(x)

        x=x.unsqueeze(-1)
        d1 = self.down1(x)
        

        d1 = self.res1(d1)
       
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        upd1=d1
        d1 = d1.squeeze(-1)
        d11=d1
        
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        
        d2 = self.res2(d2)


        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        upd2=d2
        d2 = d2.squeeze(-1)
        d22=d2
        
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)


        #print(x.shape)
        x=torch.cat((x,t),dim=-1)

        x=x.unsqueeze(-1)
        x=self.up0(x)
        x=self.res0(x)
        x = x.squeeze(-1)
        x=self.mid0(x)
        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1 = ax1.squeeze(-1)
        ax1= self.midd(ax1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        #u1 = u1.unsqueeze(-1)
        #u1 = self.attn3(u1)
        #u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        #u2 = u2.unsqueeze(-1)
        #u2 = self.attn4(u2)
        #u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2 = u2.unsqueeze(-1)
        u2= self.up3(u2)
        u2= self.res7(u2)
        #u2 = u2.unsqueeze(-1)
        #u2= self.attn5(u2)
        #u2=u2.squeeze(-1)
        u2=u2.squeeze(-1)

        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)
        upd1=self.upd1(upd1)
        upd1=self.updres1(upd1)
        upd1=upd1.squeeze(-1)
        upd1=self.nupd1(upd1)
        
        upd2=self.upd2(upd2)
        upd2=self.updres2(upd2)
        upd2=upd2.squeeze(-1)
        upd2=self.nupd2(upd2)
        upd2=upd2.unsqueeze(-1)
        upd2=self.updd2(upd2)
        upd2=self.upddres2(upd2)
        upd2=upd2.squeeze(-1)
        upd2=self.nupdd2(upd2)
        u2=u2.squeeze(-1)

        u2=torch.cat((u2,upd1),dim=-1)
        u2=torch.cat((u2,upd2),dim=-1)
        u2=self.nn7(u2)
        u2=u2.unsqueeze(-1)
        # Final output
        u3=self.final_conv(u2).squeeze(-1)
        #print(u3.shape)
        return u3
class UNet0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.mid0 = nn.Linear(70,35)
        self.midd = nn.Linear(35,70)
        self.attn = Attention(dim=35)
        self.up0 = UpsamplingBlock(70,70)
        self.res0 = ResNetBlock(70,70)
        self.nn4 = nn.Linear(70,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.up3 = UpsamplingBlock(128,128)
        self.res7 = ResNetBlock(128,128)
        self.attn5 = Attention(dim=128)
        self.nn6= nn.Linear(64,out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        xx = x
        x=self.nn1(x)

        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)


        #print(x.shape)
        x=torch.cat((x,t),dim=-1)

        x=x.unsqueeze(-1)
        x=self.up0(x)
        x=self.res0(x)
        x = x.squeeze(-1)
        x=self.mid0(x)
        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1 = ax1.squeeze(-1)
        ax1= self.midd(ax1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        #u1 = u1.unsqueeze(-1)
        #u1 = self.attn3(u1)
        #u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        #u2 = u2.unsqueeze(-1)
        #u2 = self.attn4(u2)
        #u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2 = u2.unsqueeze(-1)
        u2= self.up3(u2)
        u2= self.res7(u2)
        #u2 = u2.unsqueeze(-1)
        #u2= self.attn5(u2)
        #u2=u2.squeeze(-1)
        u2=u2.squeeze(-1)

        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)

        # Final output
        u3=self.final_conv(u2).squeeze(-1)
        #print(u3.shape)
        return u3
class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.Lattn1 =LinearAttention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.Lattn2 =LinearAttention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.attn = Attention(dim=35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.Lattn3 =LinearAttention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.Lattn4 = LinearAttention(dim=64)
        self.nn6= nn.Linear(64,out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        x=self.nn1(x)
        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = self.Lattn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2 = self.Lattn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)


        #print(x.shape)
        x=x+t

        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = self.Lattn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 = self.Lattn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)

        # Final output
        u3=self.final_conv(u2).squeeze(-1)
        #print(u3.shape)
        return u3

class UNet3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.attn = Attention(dim=35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.attn00 = Attention(dim=35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.nn6= nn.Linear(64,out_channels)

        self.up3 =UpsamplingBlock(out_channels,out_channels)
        self.res7=ResNetBlock(out_channels,out_channels)
        self.attn5 =Attention(dim=out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        x=self.nn1(x)
        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)



        #print(x.shape)
        x=x+t

        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1=self.up0(ax1)
        ax1=self.res0(ax1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)
        u3= self.up3(u2)
        u3= self.res7(u3)
        u3=u3.unsqueeze(-1)
        u3= self.attn5(u3)
        u3=u3.squeeze(-1)
        # Final output
        u3=self.final_conv(u3).squeeze(-1)
        #print(u3.shape)
        return u3

class UNet3_ITER(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.attn = Attention(dim=35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.attn00 = Attention(dim=35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.nn6= nn.Linear(64,out_channels)

        self.up3 =UpsamplingBlock(out_channels,out_channels)
        self.res7=ResNetBlock(out_channels,out_channels)
        self.attn5 =Attention(dim=out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        x=self.nn1(x)
        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)



        #print(x.shape)
        x=x+t

        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1=self.up0(ax1)
        ax1=self.res0(ax1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)
        u3= self.up3(u2)
        u3= self.res7(u3)
        u3=u3.unsqueeze(-1)
        u3= self.attn5(u3)
        u3=u3.squeeze(-1)
        # Final output
        u3=self.final_conv(u3).squeeze(-1)
        #print(u3.shape)
        return u3

class UNet4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.n0 = nn.Linear(35,175)
        self.n00 = nn.Linear(175,35)
        self.attn = Attention(dim=35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.attn00 = Attention(dim=35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.nn6= nn.Linear(64,out_channels)

        self.up3 =UpsamplingBlock(out_channels,out_channels)
        self.res7=ResNetBlock(out_channels,out_channels)
        self.attn5 =Attention(dim=out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        x=self.nn1(x)
        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)
        x=self.n0(x)



        #print(x.shape)
        x=x+t


        x=self.n00(x)
        #x=self.n00(x)
        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1=self.up0(ax1)
        ax1=self.res0(ax1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)
        u3= self.up3(u2)
        u3= self.res7(u3)
        u3=u3.unsqueeze(-1)
        u3= self.attn5(u3)
        u3=u3.squeeze(-1)
        # Final output
        u3=self.final_conv(u3).squeeze(-1)
        #print(u3.shape)
        return u3


class UNet5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,256)
        self.down22 =DownsamplingBlock(256,256)
        self.res22 = ResNetBlock(256,256)
        self.attn22 =Attention(dim=256)
        self.nn22 = nn.Linear(256,512)
        self.down33 =DownsamplingBlock(512,512)
        self.res33 = ResNetBlock(512,512)
        self.attn34 =Attention(dim=512)
        self.nn33 =nn.Linear(512,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.attn01=Attention(35)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.n0 = nn.Linear(35,175)
        self.n00 = nn.Linear(175,35)
        self.attn = Attention(dim=70)
        self.mid11 =nn.Linear(70,35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.attn00 = Attention(dim=35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,256)
        self.up22 = UpsamplingBlock(256,256)
        self.res22 =ResNetBlock(256,256)
        self.attn33 =Attention(256)
        self.nn55 = nn.Linear(256,512)
        self.up33 =UpsamplingBlock(512,512)
        self.res66 =ResNetBlock(512,512)
        self.attn44 =Attention(dim=512)
        self.nn44 =nn.Linear(1024,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.nn6= nn.Linear(64,out_channels)

        self.up3 =UpsamplingBlock(out_channels,out_channels)
        self.res7=ResNetBlock(out_channels,out_channels)
        self.attn5 =Attention(dim=out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        x=self.nn1(x)
        #xx=x
        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down22(d2)
        d3 = self.res22(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn22(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn22(d3)
        d33=d3
        d3=d3.unsqueeze(-1)
        d4=self.down33(d3)
        d4=self.res33(d4)
        d4=d4.unsqueeze(-1)
        #print(d4.shape)
        d4=self.attn34(d4)
        d4=d4.squeeze(-1)

        #x=d4.squeeze(-1)
        #print(d4.shape)
        d4=d4.squeeze(-1)
        d55= d4
        d5=self.nn33(d4)
        d5=d5.unsqueeze(-1)
        d5=self.down3(d5)
        d5=self.res3(d5)
        d5=d5.unsqueeze(-1)
        d5=self.attn0(d5)
        d5=d5.squeeze(-1)
        d5=d5.squeeze(-1)
        d6=self.nn0(d5)
        d6=d6.unsqueeze(-1)
        d6=self.down4(d6)
        d6=self.res4(d6)
        d6=d6.unsqueeze(-1)
        d6=self.attn01(d6)
        d6=d6.squeeze(-1)
        x=d6.squeeze(-1)
        #x=self.n0(x)


        #print(x.shape)
        x=torch.cat((x,t),dim=1)



        #x=self.n00(x)
        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1=ax1.squeeze(-1)
        ax1=self.mid11(ax1)
        ax1=ax1.unsqueeze(-1)

        ax1=self.up0(ax1)
        ax1=self.res0(ax1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        u12 = self.up22(u1)
        u12 = self.res22(u12)
        u12 = u12.unsqueeze(-1)
        u12 = self.attn33(u12)
        u12 =u12.squeeze(-1)
        u12 = u12.squeeze(-1)
        u12 = self.nn55(u12)
        u12 = u12.unsqueeze(-1)
        u13 = self.up33(u12)
        u13 = self.res66(u13)
        u13= u13.unsqueeze(-1)
        u13 = self.attn44(u13)
        u13 = u13.squeeze(-1)
        u13 = u13.squeeze(-1)
        u13= torch.cat((u13,d33),dim=1)
        u13 = self.nn44(u13)
        u13 = u13.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u13)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        #u2 = torch.cat((u2,xx),dim=1)
        u2 = u2.unsqueeze(-1)
        u3= self.up3(u2)
        u3= self.res7(u3)
        u3=u3.unsqueeze(-1)
        u3= self.attn5(u3)
        u3=u3.squeeze(-1)
        # Final output
        u3=self.final_conv(u3).squeeze(-1)
        #print(u3.shape)
        return u3

class UNet6(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,256)
        self.down22 =DownsamplingBlock(256,256)
        self.res22 = ResNetBlock(256,256)
        self.attn22 =Attention(dim=256)
        self.nn22 = nn.Linear(256,512)
        self.down33 =DownsamplingBlock(512,512)
        self.res33 = ResNetBlock(512,512)
        self.attn34 =Attention(dim=512)
        self.nn33 =nn.Linear(512,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.attn01=Attention(35)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.n0 = nn.Linear(35,175)
        self.n00 = nn.Linear(175,35)
        self.attn = Attention(dim=70)
        self.mid11 =nn.Linear(70,35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.attn00 = Attention(dim=35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,256)
        self.up22 = UpsamplingBlock(256,256)
        self.res22 =ResNetBlock(256,256)
        self.attn33 =Attention(256)
        self.nn55 = nn.Linear(512,512)
        self.up33 =UpsamplingBlock(512,512)
        self.res66 =ResNetBlock(512,512)
        self.attn44 =Attention(dim=512)
        self.nn44 =nn.Linear(1024,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.nn6= nn.Linear(128,out_channels)

        self.up3 =UpsamplingBlock(out_channels,out_channels)
        self.res7=ResNetBlock(out_channels,out_channels)
        self.mid3 = nn.Linear(out_channels,64)
        self.nn7 = nn.Linear(128,out_channels)
        self.attn5 =Attention(dim=out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        x=self.nn1(x)
        xx=x
        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down22(d2)
        d3 = self.res22(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn22(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d31=d3
        d3=self.nn22(d3)
        d33=d3
        d3=d3.unsqueeze(-1)
        d4=self.down33(d3)
        d4=self.res33(d4)
        d4=d4.unsqueeze(-1)
        #print(d4.shape)
        d4=self.attn34(d4)
        d4=d4.squeeze(-1)

        #x=d4.squeeze(-1)
        #print(d4.shape)
        d4=d4.squeeze(-1)
        d55= d4
        d5=self.nn33(d4)
        d5=d5.unsqueeze(-1)
        d5=self.down3(d5)
        d5=self.res3(d5)
        d5=d5.unsqueeze(-1)
        d5=self.attn0(d5)
        d5=d5.squeeze(-1)
        d5=d5.squeeze(-1)
        d6=self.nn0(d5)
        d6=d6.unsqueeze(-1)
        d6=self.down4(d6)
        d6=self.res4(d6)
        d6=d6.unsqueeze(-1)
        d6=self.attn01(d6)
        d6=d6.squeeze(-1)
        x=d6.squeeze(-1)
        #x=self.n0(x)


        #print(x.shape)
        x=torch.cat((x,t),dim=1)



        #x=self.n00(x)
        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1=ax1.squeeze(-1)
        ax1=self.mid11(ax1)
        ax1=ax1.unsqueeze(-1)

        ax1=self.up0(ax1)
        ax1=self.res0(ax1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        u12 = self.up22(u1)
        u12 = self.res22(u12)
        u12 = u12.unsqueeze(-1)
        u12 = self.attn33(u12)
        u12 =u12.squeeze(-1)
        u12 = u12.squeeze(-1)
        u12=torch.cat((u12,d31),dim=1)
        u12 = self.nn55(u12)
        u12 = u12.unsqueeze(-1)
        u13 = self.up33(u12)
        u13 = self.res66(u13)
        u13= u13.unsqueeze(-1)
        u13 = self.attn44(u13)
        u13 = u13.squeeze(-1)
        u13 = u13.squeeze(-1)
        u13= torch.cat((u13,d33),dim=1)
        u13 = self.nn44(u13)
        u13 = u13.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u13)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2=self.mid2(u2)
        u2 = self.nn6(u2)

        u2 = u2.unsqueeze(-1)
        u3= self.up3(u2)
        u3= self.res7(u3)
        u3 = u3.squeeze(-1)
        u3 = self.mid3(u3)
        u3 = torch.cat((u3,xx),dim=1)
        u3 = self.nn7(u3)
        u3 = u3.unsqueeze(-1)

        u3=u3.unsqueeze(-1)
        u3= self.attn5(u3)
        u3=u3.squeeze(-1)
        # Final output
        u3=self.final_conv(u3).squeeze(-1)
        #print(u3.shape)
        return u3
class TransformerModel(nn.Module):
    def __init__(self, in_channels, nhead=5, num_encoder_layers=2, dim_feedforward=64, dropout=0.1):
        super(TransformerModel, self).__init__()
        # Transformer U-net
        # Define the transformer layer
        self.transformer_layer = nn.Transformer(
            d_model=in_channels,  # input size (same as in_channels in Linear layer)
            nhead=nhead,  # number of attention heads
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,  # For encoder-decoder architecture, specify decoder layers
            dim_feedforward=dim_feedforward,  # corresponds to the hidden dimension in feedforward layers
            dropout=dropout
        )
        self.transformer_layer2 = nn.Transformer(
            d_model=in_channels,  # input size (same as in_channels in Linear layer)
            nhead=nhead,  # number of attention heads
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,  # For encoder-decoder architecture, specify decoder layers
            dim_feedforward=dim_feedforward,  # corresponds to the hidden dimension in feedforward layers
            dropout=dropout
        )

       

        self.transformer_layer3 = nn.Transformer(
            d_model=in_channels*2,  # input size (same as in_channels in Linear layer)
            nhead=nhead,  # number of attention heads
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,  # For encoder-decoder architecture, specify decoder layers
            dim_feedforward=dim_feedforward,  # corresponds to the hidden dimension in feedforward layers
            dropout=dropout
        )
       
        self.transformer_layer4 = nn.Transformer(
            d_model=in_channels*3,  # input size (same as in_channels in Linear layer)
            nhead=nhead,  # number of attention heads
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,  # For encoder-decoder architecture, specify decoder layers
            dim_feedforward=dim_feedforward,  # corresponds to the hidden dimension in feedforward layers
            dropout=dropout
        )
 
        # You may still need to apply a final linear projection
        self.fc_out = nn.Linear(in_channels, 35)  # Final linear layer to reduce to 64 dims

    def forward(self, src, tgt):
        # src is the input sequence
        # tgt is the target sequence (can be None if only using the encoder)
        output = self.transformer_layer(src,tgt)
        #down1= output
        #output = self.transformer_layer2(output,condition)
        #down2 =output
        #output = output+tgt
        #output =torch.cat((output,down1),dim=-1)
        #c=torch.cat((v,c),dim=-1)
        #output = self.transformer_layer3(output,c)
        #output = torch.cat((output,down2),dim=-1)

        #output2 = self.transformer_layer2(src2,tgt)
        #output3 = self.transformer_layer3(src3,tgt)
        #Final_output = torch.cat((output,output2),dim=-1)
        #Final_output = torch.cat((Final_output,output3),dim=-1)
        #Final_output=  self.transformer_layer4(output,c)
        # Final projection (if needed)
        output = self.fc_out(output)
        return output
class UNet7(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.trans1=TransformerModel(in_channels=35)
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,256)
        self.down22 =DownsamplingBlock(256,256)
        self.res22 = ResNetBlock(256,256)
        self.attn22 =Attention(dim=256)
        self.nn22 = nn.Linear(256,512)
        self.down33 =DownsamplingBlock(512,512)
        self.res33 = ResNetBlock(512,512)
        self.attn34 =Attention(dim=512)
        self.nn33 =nn.Linear(512,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.attn01=Attention(35)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.n0 = nn.Linear(35,175)
        self.n00 = nn.Linear(175,35)
        self.attn = Attention(dim=70)
        self.mid11 =nn.Linear(70,35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.attn00 = Attention(dim=35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,256)
        self.up22 = UpsamplingBlock(256,256)
        self.res22 =ResNetBlock(256,256)
        self.attn33 =Attention(256)
        self.nn55 = nn.Linear(256,512)
        self.up33 =UpsamplingBlock(512,512)
        self.res66 =ResNetBlock(512,512)
        self.attn44 =Attention(dim=512)
        self.nn44 =nn.Linear(1024,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.nn6= nn.Linear(64,out_channels)

        self.up3 =UpsamplingBlock(out_channels,out_channels)
        self.res7=ResNetBlock(out_channels,out_channels)
        self.attn5 =Attention(dim=out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        x=self.nn1(x)

        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down22(d2)
        d3 = self.res22(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn22(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn22(d3)
        d33=d3
        d3=d3.unsqueeze(-1)
        d4=self.down33(d3)
        d4=self.res33(d4)
        d4=d4.unsqueeze(-1)
        #print(d4.shape)
        d4=self.attn34(d4)
        d4=d4.squeeze(-1)

        #x=d4.squeeze(-1)
        #print(d4.shape)
        d4=d4.squeeze(-1)
        d55= d4
        d5=self.nn33(d4)
        d5=d5.unsqueeze(-1)
        d5=self.down3(d5)
        d5=self.res3(d5)
        d5=d5.unsqueeze(-1)
        d5=self.attn0(d5)
        d5=d5.squeeze(-1)
        d5=d5.squeeze(-1)
        d6=self.nn0(d5)
        d6=d6.unsqueeze(-1)
        d6=self.down4(d6)
        d6=self.res4(d6)
        d6=d6.unsqueeze(-1)
        d6=self.attn01(d6)
        d6=d6.squeeze(-1)
        x=d6.squeeze(-1)
        #x=self.n0(x)


        #print(x.shape)
        x=torch.cat((x,t),dim=1)



        #x=self.n00(x)
        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1=ax1.squeeze(-1)
        ax1=self.mid11(ax1)
        ax1=ax1.unsqueeze(-1)

        ax1=self.up0(ax1)
        ax1=self.res0(ax1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        u12 = self.up22(u1)
        u12 = self.res22(u12)
        u12 = u12.unsqueeze(-1)
        u12 = self.attn33(u12)
        u12 =u12.squeeze(-1)
        u12 = u12.squeeze(-1)
        u12 = self.nn55(u12)
        u12 = u12.unsqueeze(-1)
        u13 = self.up33(u12)
        u13 = self.res66(u13)
        u13= u13.unsqueeze(-1)
        u13 = self.attn44(u13)
        u13 = u13.squeeze(-1)
        u13 = u13.squeeze(-1)
        u13= torch.cat((u13,d33),dim=1)
        u13 = self.nn44(u13)
        u13 = u13.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u13)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)
        u3= self.up3(u2)
        u3= self.res7(u3)
        u3=u3.unsqueeze(-1)
        u3= self.attn5(u3)
        u3=u3.squeeze(-1)
        # Final output
        u3=self.final_conv(u3).squeeze(-1)
        #print(u3.shape)
        return u3
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv2(x)
        #print(x.shape)
        #print(residual.shape)
        return x + residual
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, out_channels)
        self.gcn2 = GCNConv(out_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        residual = x
        x = self.gcn1(x, edge_index)
        x = self.norm(x)
        x = self.activation(x)
        x = self.gcn2(x, edge_index)
        return x + residual

class UNet8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.gcn1 =GCNBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.gcn2 = GCNBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.gcn3 = GCNBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.gcn4 = GCNBlock(35,35)
        self.attn = Attention(dim=35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.gcn0 = GCNBlock(35,35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.gcn5 = GCNBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.gcn6 = GCNBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.up3 = UpsamplingBlock(128,128)
        self.res7 = ResNetBlock(128,128)
        self.gcn7 = GCNBlock(128,128)
        self.attn5 = Attention(dim=128)
        self.nn6= nn.Linear(64,out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t,edge):
        # Downsampling
        xx = x
        x=self.nn1(x)

        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1= self.gcn1(d1,edge)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)
        d2= self.gcn2(d2,edge)
        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3 = self.gcn3(d3,edge)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        d4= self.gcn4(d4,edge)
        x=d4.squeeze(-1)


        #print(x.shape)
        x=x+t

        x=x.unsqueeze(-1)
        x=self.up0(x)
        x=self.res0(x)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = self.gcn5(u1,edge)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        u2 = self.gcn6(u2,edge)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2 = u2.unsqueeze(-1)
        u2= self.up3(u2)
        u2= self.res7(u2)
        u2 = self.gcn7(u2,edge)
        u2 = u2.unsqueeze(-1)
        u2= self.attn5(u2)
        u2=u2.squeeze(-1)
        u2=u2.squeeze(-1)

        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)

        # Final output
        u3=self.final_conv(u2).squeeze(-1)
        #print(u3.shape)
        return u3


# Denoising Diffusion Model
class DenoiseDiffusion(nn.Module):
    def __init__(self,input_size,output_size,noise_steps):
        super().__init__()
        #self.input_size = input_size
        self.input_size= input_size
        self.output_size = output_size
        self.noise_steps = noise_steps
        self.hidden_size = 512
        self.time_dim=70
        self.time_dim2=35
        #self.time_dim2=self.time_sim
        #self.time_dim = self.hidden_size
        self.gaussiandiffusion = GaussianDiffusion(self.input_size,self.noise_steps,self.output_size)
        self.timeembedding = TimeEmbedding(self.time_dim)
        self.betas = self.gaussiandiffusion.beta
        self.alpha = 1-self.betas
        self.alpha_bar = torch.cumprod(self.alpha,dim=0)
        #self.d1 = DownsamplingBlock2(self.input_size,self.input_size)
        #print(self.input_size)
        #self.nn1 =nn.Linear(self.output_size,self.output_size*5)
        self.fc1 = nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size+self.time_dim2,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size+self.time_dim2,self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size + self.time_dim2, self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size + self.time_dim2, self.hidden_size)
        self.fc6 = nn.Linear(self.hidden_size + self.time_dim2, self.hidden_size)
        #self.fc6 = nn.Linear(self.hidden_size + self.time_dim2, self.hidden_size)
        self.fc7 = nn.Linear(self.hidden_size+self.time_dim2,self.output_size)
        #self.up1 = UpsamplingBlock2(self.input_size,self.input_size)
        self.mean=0
        self.var=0

    def q_xt_x0(self,x0,t):
        mean = self.gaussiandiffusion.alpha_hat
        mean = self.gaussiandiffusion.alpha_hat[t]**0.5*x0
        var = 1-self.gaussiandiffusion.alpha_hat[t]
        self.mean=mean
        self.var=var
        return mean , var

    def q_sample(self,x0,t,eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean,var = self.q_xt_x0(x0,t)
        return mean
        #return mean+(var**0.5)*eps

    def p_sample(self,xt,t):
    #def p_sample(self,xt,t):
        #eps_theta=self.forward(xt,t)
        eps_theta = self.forward(xt,t)
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[t]
        alpha = self.gaussiandiffusion.alpha[t]

        eps_coef = (1-alpha)/(1-alpha_hat)**0.5
        #eps_coef2=eps_coef.repeat(1,5)
        #alpha2=alpha.repeat(1,5)
        mean = 1/(alpha**0.5)*(xt-eps_coef*eps_theta)
        var = self.gaussiandiffusion.beta[t]
        #var2=var.repeat(1,5)
        eps = torch.randn_like(xt)

        return mean + (var**0.5)*eps

    def forward(self,xt,t):

        t = self.timeembedding(t)
        #xt=xt.unsqueeze(-1)
        #xt=self.d1(xt)
        #xt=self.up1(xt)
        #xt=xt.squeeze(-1)

        emb = self.fc1(xt)
        emb = torch.cat((emb,t),dim=1)
        emb = F.elu(self.fc2(emb))
        emb = torch.cat((emb,t),dim=1)
        emb = F.elu(self.fc3(emb))
        emb = torch.cat((emb,t),dim=1)
        emb = F.elu(self.fc4(emb))
        emb = torch.cat((emb, t), dim=1)
        emb = F.elu(self.fc5(emb))
        emb = torch.cat((emb, t), dim=1)
        emb = F.elu(self.fc6(emb))
        emb = torch.cat((emb, t), dim=1)
        emb = F.elu(self.fc7(emb))
        return emb
class DenoiseDiffusion2(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0,t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_= self.forward(xt,t)
        tt=torch.tensor(0).long().to(device="cpu")
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean
        return mean + (var ** 0.5) * eps

    def forward(self, x,t):

        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        mean,var=self.q_xt_x0(x,t)
        return x,mean,var

class DenoiseDiffusion4(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean + (var ** 0.5) * eps

    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var


class DenoiseDiffusion5(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(t).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean + (var ** 0.5) * eps

    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var
class DenoiseDiffusion6(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet2(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(t).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean + (var ** 0.5) * eps

    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var

class DenoiseDiffusion7(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet3(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(t).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean + (var ** 0.5) * eps

    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var
class DenoiseDiffusion8(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet3(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)
        self.mean=0
        self.logvar=0

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(t).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean + (var ** 0.5) * eps

    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var

class DenoiseDiffusion77(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet3(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)

    def q_xt_x0(self, x0, t):
        #gaussian 을  t 에 맟추어서 분포시
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(t).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean + (var ** 0.5) * eps

    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var
class DenoiseDiffusion9(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet4(175, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)
        self.mean=0
        self.logvar=0
        self.n= nn.Linear(35,input_size)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean + (var ** 0.5) * eps

    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var


from tqdm import tqdm
class DenoiseDiffusion10(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet4(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        self.n= nn.Linear(35,input_size)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(t).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)

        return mean + (var ** 0.5) * eps


    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var
class DenoiseDiffusion11(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet5(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        self.n= nn.Linear(35,input_size)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)

        return mean + (var ** 0.5) * eps


    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var
class DenoiseDiffusion12(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet4(175, 175)  # Assuming 3 input and 3 output channels for RGB
        self.encoder=Encoder(input_size,175)
        self.decoder=Decoder(350,350,5,output_size*5)
        self.time_embed = TimeEmbedding(350)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        self.nn1= nn.Linear(175,output_size)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)

        x=mean + (var ** 0.5) * eps
        x=self.nn1(x)
        return x


    def forward(self, x,t,c):
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)

        return self.decoder(zc,c),mean,var

class DenoiseDiffusion13(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.trans=TransformerModel(in_channels=input_size)
        self.unet = UNet3(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        self.n= nn.Linear(35,input_size)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)

        return mean + (var ** 0.5) * eps


    def forward(self, x,t,c):
        xc=self.trans(x,c)
        z,mu,logvar=self.encoder(xc,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)

        return self.decoder(zc,c),mean,var

class DenoiseDiffusion13_ITER(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.trans=TransformerModel(in_channels=input_size)
        self.unet = UNet4(35, 175)  # Assuming 3 input and 3 output channels for RGB
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,210,5,5*output_size)
        self.decoder2 = Decoder(35,175,5,output_size)
        self.time_embed = TimeEmbedding(350)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        self.n= nn.Linear(35,input_size)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        z = mean+(var**0.5)*eps
        return self.decoder2(z,c)
        #return mean + (var ** 0.5) * eps


    def forward(self, x,t,c):
        #xc=self.trans(x,c)
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)

        return self.decoder(zc,c),mean,var

class DenoiseDiffusion14(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        #self.decoder2 = Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0

    def generate(self,z,c):
        z=torch.cat((z,c),dim=-1)
        return self.decoder(z,c)
    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean
        #return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        #return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
        return mean
    def forward(self, x,t,c):
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        return self.decoder(zc,c),mean,var

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class ActNorm(nn.Module):
    """Activation Normalization Layer."""
    def __init__(self, num_channels):
        super().__init__()
        self.bias = Parameter(torch.zeros(1, num_channels, 1, 1))
        self.scale = Parameter(torch.ones(1, num_channels, 1, 1))
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                mean = x.mean(dim=(0, 2, 3), keepdim=True)
                std = x.std(dim=(0, 2, 3), keepdim=True)
                self.bias.data.copy_(-mean)
                self.scale.data.copy_(1 / (std + 1e-6))
            self.initialized = True
        return self.scale * (x + self.bias)

class Invertible1x1Conv(nn.Module):
    """Invertible 1x1 Convolution."""
    def __init__(self, num_channels):
        super().__init__()
        weight = torch.qr(torch.randn(num_channels, num_channels))[0]
        self.weight = Parameter(weight)
    
    def forward(self, x, reverse=False):
        b, c, h, w = x.size()
        log_det = torch.slogdet(self.weight)[1] * h * w
        if reverse:
            weight_inv = torch.inverse(self.weight)
            return F.conv2d(x, weight_inv.unsqueeze(-1).unsqueeze(-1)), -log_det
        else:
            return F.conv2d(x, self.weight.unsqueeze(-1).unsqueeze(-1)), log_det

class AffineCoupling(nn.Module):
    """Affine Coupling Layer."""
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        params = self.net(x1)
        scale, shift = params.chunk(2, dim=1)
        scale = torch.sigmoid(scale + 2.0)  # Scale stabilization
        if reverse:
            x2 = (x2 - shift) / scale
        else:
            x2 = scale * x2 + shift
        return torch.cat([x1, x2], dim=1)

class GlowStep(nn.Module):
    """One step of Glow."""
    def __init__(self, num_channels):
        super().__init__()
        self.actnorm = ActNorm(num_channels)
        self.inv_conv = Invertible1x1Conv(num_channels)
        self.coupling = AffineCoupling(num_channels)

    def forward(self, x, reverse=False):
        if reverse:
            x = self.coupling(x, reverse=True)
            x, _ = self.inv_conv(x, reverse=True)
            x = self.actnorm(x)
        else:
            x = self.actnorm(x)
            x, log_det1 = self.inv_conv(x)
            x = self.coupling(x)
        return x

class Glow(nn.Module):
    """Glow Model."""
    def __init__(self, num_channels, num_levels, num_steps):
        super().__init__()
        self.levels = nn.ModuleList()
        for _ in range(num_levels):
            steps = nn.ModuleList([GlowStep(num_channels) for _ in range(num_steps)])
            self.levels.append(steps)
    
    def forward(self, x, reverse=False):
        log_det = 0
        if reverse:
            for steps in reversed(self.levels):
                for step in reversed(steps):
                    x = step(x, reverse=True)
        else:
            for steps in self.levels:
                for step in steps:
                    x = step(x)
        return x
# Define the ODE Function

class DenoiseDiffusion14_mul(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size*2,35)
        self.encoder2 =Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.decoder2=Decoder(35,70,5,output_size)
        self.flow = nn.Linear(70,64)
        self.flow2 =nn.Linear(64,35)
        self.plane1=nn.Linear(105,100)
        self.plane2=nn.Linear(100,70)
        self.plane3=nn.Linear(70,35)
        self.h0 = torch.ones(2,2048, 35)
        #self.decoder2 = Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0

    def generate(self,z,c):
        z=torch.cat((z,c),dim=-1)
        return self.decoder(z,c)
    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,v,x0,x1,x2,t,c):
        eps_theta,_,_ = self.forward(xt,v,x0,x1,x2,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        #return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
        return mean
    def forward(self, x,v,x0,x1,x2,t,c):
        flow=torch.cat((x,v),dim=-1)
        flow_box=self.flow(flow)
        flow_box=self.flow2(flow_box)
        plane=torch.cat((x0,x1),dim=-1)
        plane=torch.cat((plane,x2),dim=-1)
        plane=self.plane1(plane)
        plane=self.plane2(plane)
        plane=self.plane3(plane)
        flow_op=torch.cat((flow_box,plane),dim=-1)

        z,mu,logvar=self.encoder(flow_op,c)
        #z2=self.trans(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #x2 = self.unet(z2,t_emb).to(device="cpu") 
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        #zc2=torch.cat((x2,z),dim=-1)
        return self.decoder(zc,c),mean,var

import torch
import torch.nn as nn
import torchdiffeq  # Library for ODE solvers

# Define ODE function
class ODEfunc(nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t,h):
        return self.fc(h)

# Neural ODE model
class NeuralODE(nn.Module):
    def __init__(self, dim):
        super(NeuralODE, self).__init__()
        self.ode_func = ODEfunc(dim)
    
    def forward(self, h0, t):
        # Solve ODE using odeint
        h = torchdiffeq.odeint(self.ode_func, h0, t)
        return h

class DenoiseDiffusion14_mul_vec(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size*3,35)
        self.encoder2 =Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.decoder2=Decoder(35,70,5,output_size)
        self.flow = nn.Linear(70,64)
        self.flow2 =nn.Linear(64,35)
        self.plane1=nn.Linear(70,50)
        self.plane2=nn.Linear(50,35)
        #self.plane3=nn.Linear(70,35)
        self.h0 = torch.ones(2,2048, 35)
        self.func=ODEfunc(dim=35)
        self.ode = NeuralODE(dim=35)
        self.attn = Attention(dim=35)
        self.attn2 = Attention(dim=35)
        #self.decoder2 = Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        self.dct_basis = self.create_dct_basis(35)  # Dimension 35
        self.idct_basis = self.create_idct_basis(35)

    def create_dct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        dct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)
        return dct_basis

    def create_idct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        idct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)
        return idct_basis

    def dct(self, x):
        # Apply DCT transformation
        return torch.matmul(x, self.dct_basis)

    def idct(self, x):
        # Apply IDCT transformation
        return torch.matmul(x, self.idct_basis)

    def generate(self,z,c):
        z=torch.cat((z,c),dim=-1)
        return self.decoder(z,c)
    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var
    def get_normal(self,x1,x2,x3):
        v1=x1-x2
        v2=x1-x3
        normal=torch.cross(v1,v2,dim=-1)
        if torch.norm(normal)<1e-4:
            eps=torch.randn(2048,35)
            normal=normal+eps
        return normal
    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,v,tim,x1,x2,t,c):
        eps_theta,_,_,_,_= self.forward(xt,v,tim,x1,x2,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        #return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
        return mean
    def forward(self, x,v,tim,x1,x2,t,c):
        time_grid = torch.linspace(0, 1, steps=10)
        y0 = v # Extend v to shape [2048, 35]
        #y0=self.dct(y0)
        vt=self.ode(y0,time_grid)
        #vt=self.idct(vt)
        vtx=vt[9,::]-vt[0,::]
        
                 # Time grid
        flow=torch.cat((x,v),dim=-1)
        #flow=torch.cat((flow,tim),dim=-1)

        flow_box=self.flow(flow)
        flow_box=self.flow2(flow_box)
        flow_box=flow_box.unsqueeze(-1)
        flow_box=flow_box.unsqueeze(-1)
        flow_box =self.attn(flow_box)
        flow_box=flow_box.squeeze(-1)
        flow_box=flow_box.squeeze(-1)
        #plane=torch.cat((x0,x1),dim=-1)
        plane=torch.cat((x1,x2),dim=-1)
        plane=self.plane1(plane)
        plane=self.plane2(plane)
        plane=plane.unsqueeze(-1)
        plane=plane.unsqueeze(-1)
        plane = self.attn2(plane)
        plane=plane.squeeze(-1)
        plane=plane.squeeze(-1)
        #plane=self.plane3(plane)

        #eps=torch.randn(2048,35)
        #eps=eps*0.0000000001
        #if torch.norm(plane)<1e-9:
        #    plane=plane+eps
        
        #plane=self.get_normal(x0,x1,x2)
        #plane=self.plane3(plane)
        flow_op=torch.cat((flow_box,plane),dim=-1)
        flow_box_t=torch.cat((flow_op,vtx),dim=-1)


        z,mu,logvar=self.encoder(flow_box_t,c)
        #z2=self.trans(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #x2 = self.unet(z2,t_emb).to(device="cpu") 
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        #zc2=torch.cat((x2,z),dim=-1)
        return self.decoder(zc,c),mean,var,vt[0,::],vt[9,::]
class ComplexConv1d(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.real_conv = torch.nn.Conv1d(*args, **kwargs)
        self.imag_conv = torch.nn.Conv1d(*args, **kwargs)

    def forward(self, x):
        real = x.real
        imag = x.imag

        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)

        return torch.complex(real_out, imag_out)
class DenoiseDiffusion14_mul_vec_n_atten(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNetTransformer(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size*3,35)
        self.encoder2 =Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.decoder2=Decoder(35,70,5,output_size)
        self.flow = nn.Linear(105,64)
        self.flow2 =nn.Linear(64,35)
        self.plane1=nn.Linear(70,50)
        self.plane2=nn.Linear(50,35)
        #self.plane3=nn.Linear(70,35)
        self.h0 = torch.ones(2,2048, 35)
        self.func=ODEfunc(dim=35)
        self.ode = NeuralODE(dim=35)
        self.ode2 = NeuralODE(dim=35)
        self.attn = Attention(dim=35)
        self.attn2 = Attention(dim=35)
        self.conv = ComplexConv1d(in_channels=1, out_channels=1, kernel_size=1)
        #self.decoder2 = Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        self.dct_basis = self.create_dct_basis(35)  # Dimension 35
        self.idct_basis = self.create_idct_basis(35)

    def create_dct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        dct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)
        return dct_basis

    def create_idct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        idct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)
        return idct_basis

    def dct(self, x):
        # Apply DCT transformation
        return torch.matmul(x, self.dct_basis)

    def idct(self, x):
        # Apply IDCT transformation
        return torch.matmul(x, self.idct_basis)

    def generate(self,z,c):
        z=torch.cat((z,c),dim=-1)
        return self.decoder(z,c)
    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var
    def get_normal(self,x1,x2,x3):
        v1=x1-x2
        v2=x1-x3
        normal=torch.cross(v1,v2,dim=-1)
        if torch.norm(normal)<1e-4:
            eps=torch.randn(2048,35)
            normal=normal+eps
        return normal
    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,v,x1,x2,t,c):
        eps_theta,_,_,_,_= self.forward(xt,v,x1,x2,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        #return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
        return mean
    def forward(self, x,v,x1,x2,t,c):
        time_grid = torch.linspace(0, 1, steps=10)
        time_grid2 = torch.linspace(0,1,steps=2)
        y0 = v # Extend v to shape [2048, 35]
        #y0=torch.fft.fft2(y0)
        #print(y0.shape)
        #y0=y0.unsqueeze(-1)
        #y0=self.conv(y0.permute(0,2,1))
        #y0=y0.permute(0,2,1)
        #y0=y0.squeeze(-1)

        #y0=torch.fft.ifft2(y0)
        #y0=y0.real
        vt=self.ode(y0,time_grid)
        #vt=self.idct(vt)
        vtx=vt[9,::]-vt[0,::]
        riman=self.ode(x,time_grid2)
        riman_d=riman[1,::].mul(riman[0,::])
                 # Time grid
        flow=torch.cat((x,v),dim=-1)
        flow=torch.cat((flow,riman_d),dim=-1)
         
        flow_box=self.flow(flow)
        flow_box=self.flow2(flow_box)
        #flow_box=flow_box.unsqueeze(-1)
        #flow_box=flow_box.unsqueeze(-1)
        #flow_box =self.attn(flow_box)
        #flow_box=flow_box.squeeze(-1)
        #flow_box=flow_box.squeeze(-1)
        #plane=torch.cat((x0,x1),dim=-1)
        plane=torch.cat((x1,x2),dim=-1)
        plane=self.plane1(plane)
        plane=self.plane2(plane)
        #plane=plane.unsqueeze(-1)
        #plane=plane.unsqueeze(-1)
        #plane = self.attn2(plane)
        #plane=plane.squeeze(-1)
        #plane=plane.squeeze(-1)
        #plane=self.plane3(plane)

        #eps=torch.randn(2048,35)
        #eps=eps*0.0000000001
        #if torch.norm(plane)<1e-9:
        #    plane=plane+eps
        
        #plane=self.get_normal(x0,x1,x2)
        #plane=self.plane3(plane)
        flow_op=torch.cat((flow_box,plane),dim=-1)
        flow_box_t=torch.cat((flow_op,vtx),dim=-1)


        z,mu,logvar=self.encoder(flow_box_t,c)
        #z2=self.trans(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb,flow_box).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #x2 = self.unet(z2,t_emb).to(device="cpu") 
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        #zc2=torch.cat((x2,z),dim=-1)
        return self.decoder(zc,c),mean,var,vt[0,::],vt[9,::]
import torch
import torch.nn as nn
import math

class DenoiseDiffusion14_dct(nn.Module):
    def __init__(self, input_size, noise_steps, latent_size, output_size):
        super().__init__()
        self.timesteps = noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming input and output dimensions of 35 for U-Net
        self.encoder = Encoder(input_size, 35)
        self.decoder = Decoder(35, 70, 5, output_size)
        self.trans = TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta = self.gaussiandiffusion.beta
        self.mean = 0
        self.logvar = 0

        # DCT and IDCT Basis
        self.dct_basis = self.create_dct_basis(35)  # Dimension 35
        self.idct_basis = self.create_idct_basis(35)

    def create_dct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        dct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)
        return dct_basis

    def create_idct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        idct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)
        return idct_basis

    def dct(self, x):
        # Apply DCT transformation
        return torch.matmul(x, self.dct_basis)

    def idct(self, x):
        # Apply IDCT transformation
        return torch.matmul(x, self.idct_basis)

    def generate(self, z, c):
        z = torch.cat((z, c), dim=-1)
        return self.decoder(z, c)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt, t, c):
        eps_theta, _, _,_ = self.forward(xt, t, c)
        tt = torch.tensor(0).long()
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean

    def forward(self, x, t, c):
        # DCT Transformation
        z, mu, logvar = self.encoder(x, c)
        z_dct = self.dct(z)  # Apply DCT to latent representation
        noise = torch.randn_like(z)
        # Time Embedding
        t_emb = self.time_embed(t)
        z_noise = self.q_sample(z_dct,t,noise)
        # Pass through U-Net

        # IDCT Transformation
        unet_output_idct = self.idct(z_noise)  # Apply IDCT after U-Net
        unet_output = self.unet(unet_output_idct, t_emb).to(device="cpu")

      

        # Decoder
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean, var = self.q_xt_x0(unet_output_idct, tt)
        zc = torch.cat((unet_output, z), dim=-1)
        return self.decoder(zc, c), mean, var,z
class DenoiseDiffusion16(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean+(var**0.5)
        #return mean
        #return mean + (var ** 0.5) * eps
    def forward(self, x,t,c):
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        for i in range(1,self.timesteps):
            x = self.unet(x,t_emb).to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        return self.decoder(zc,c),mean,var
class DenoiseDiffusion14_Loop(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet3(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,idx,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        if idx == 0:
            return mean
        else:
            return mean + (var ** 0.5) * eps
    def p_sample_loop(self,xt,t,c):
        b=xt
        img=torch.randn(35)
        imgs=[]
        for i in tqdm(reversed(range(0, self.timesteps)), desc = 'sampling loop time step', total=self.timesteps):
            img=self.p_sample(xt,t, i,c)
            #print(img.shape)
            imgs.append(img)
        return imgs
    def forward(self, x,t,c):
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        #z=self.trans(z,c)
        zc=torch.cat((x,z),dim=-1)

        return self.decoder(zc,c),mean,var


class DenoiseDiffusion15(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet0(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        #self.mixdecoder = MixedDecoder()
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        return mean
        #return mean + (var ** 0.5) * eps
    def forward(self, x,t,c):
        z,mu,logvar=self.encoder(x,c)
        z2,mu2,logvar2 = self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,t)
        zc=torch.cat((x,z),dim=-1)
        return self.decoder(z, zc), mean, var
        #return self.decoder(zc,z),mean,var
        #return self.decoder(zc,c),mean,var

class DenoiseDiffusion18(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet0(175, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,175)
        self.decoder=Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(350)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var
    def reduce_mean_to_35(self, mean_175):
        # 175차원을 5개의 그룹으로 나누고, 각 그룹의 평균을 구해 35차원으로 축소
        mean_35 = mean_175.view(mean_175.size(0), 5, -1, *mean_175.size()[2:]).mean(dim=1)
        return mean_35

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        return mean
        #return mean + (var ** 0.5) * eps
    def forward(self, x,t,c):
        z,mu,logvar=self.encoder(x,c)
        #z2,mu2,logvar2 = self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,t)
        zc=torch.cat((x,z),dim=-1)
        return self.decoder(z, zc), mean, var
        #return self.decoder(zc,z),mean,var
        #return self.decoder(zc,c),mean,var
class DenoiseDiffusion0(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet0(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        return mean
        #return mean + (var ** 0.5) * eps
    def forward(self, x,t,c):
        #z,mu,logvar=self.encoder(x,c)
        #z2,mu2,logvar2 = self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb).to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,t)
        #zc=torch.cat((x,z),dim=-1)
        #return self.decoder(z, zc), mean, var
        #return self.decoder(zc,z),mean,var
        #return self.decoder(zc,c),mean,var
        return x,mean,var

class Conv1dModel(nn.Module):
    def __init__(self):
        super(Conv1dModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

class Generator(nn.Module):
    def __init__(self,input,output):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512,output),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self,input,output):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)
class DenoiseDiffusion20(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder2(input_size,35)
        self.encoder2 = Encoder2(input_size,35)
        self.encoder3 = Encoder2(input_size,35)
        #self.encoder4 = Encoder(105,35)
        self.cnn=Conv1dModel()
        self.attn = Attention(dim=35)
        self.decoder=Decoder(35,70,5,output_size)
        #self.generator =  Generator(input,3*input)
        #self.discriminator= Discriminator(3*input,input)
        #self.mixdecoder =MixedDecoder(35,70,output_size,10,10,10)
        #self.decoder2 = Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=105)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,xt2,xt3,t,c,c2,c3):
        eps_theta,_,_,_= self.forward(xt,xt2,xt3,t,c,c2,c3)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        #return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
        return mean
    def forward(self, x,x2,x3,t,c,c2,c3):
        #cc=torch.cat((c,c2),dim=-1)
        #cc=torch.cat((cc,c3),dim=-1)
        z,mu,logvar=self.encoder(x,c)
        z2,mu2,logvar2 = self.encoder2(x2,c2)
        z3,mu3,logvar3 = self.encoder3(x3,c3)
        #z=z+z2
        #z=torch.cat((z,z2),dim=-1)
        z=torch.cat((z,z2),dim=-1)
        z=torch.cat((z,z3),dim=-1)
        tgt=torch.cat((c,c2),dim=-1)
        tgt=torch.cat((tgt,c3),dim=-1)
        z=self.trans(z,tgt)
        z_noise=z
        #z,mu4,logvar4= self.encoder4(z,c)
        #z=z.unsqueeze(-1)
        #z=z.permute(0,2,1) 
        #z=self.cnn(z)
        #z=z.squeeze(1)
        #z=z.unsqueeze(-1)
        #z=z.unsqueeze(-1)
        #z=self.attn(z)
        #z=z.squeeze(-1)
        #z=z.squeeze(-1)
        t_emb = self.time_embed(t)
        #t_e = t_emb.repeat(1,2)
        #x= self.trans(z,t_emb)
        #z=self.trans(z,t_e)

        x = self.unet(z,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        return self.decoder(zc,c),mean,var,z_noise

class Belfusion(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet0(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.encoder2 = Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
    
    def generate(self,z,c):
        z=torch.cat((z,c),dim=-1)
        return self.decoder(z,c)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var
    def reduce_mean_to_35(self, mean_175):
        # 175차원을 5개의 그룹으로 나누고, 각 그룹의 평균을 구해 35차원으로 축소
        mean_35 = mean_175.view(mean_175.size(0), 5, -1, *mean_175.size()[2:]).mean(dim=1)
        return mean_35

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_,_,output = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return eps_theta
        #return mean+(var**0.5)
        return mean
        #return mean + (var ** 0.5) * eps
        #return output
    def forward(self, x,t,c):
        z,mu,logvar=self.encoder(x,c)
        #z2,mu2,logvar2 = self.encoder2(x,c)
        #z2,mu2,logvar2 = self.encoder(x,c)
        noise = torch.randn_like(z)
        z_noise = self.q_sample(z,t,noise)
        t_emb = self.time_embed(t)
        x = self.unet(z_noise,t_emb).to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,t)
        zc=torch.cat((x,z),dim=-1)
        return x, z,mean, var,self.decoder(zc, z)
                                                            
        #return self.decoder(zc,z),mean,var
        #return self.decoder(zc,c),mean,var
    
class Dif_gan(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet0(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.unet2 = UNet0(35,35)
        self.encoder=Encoder(input_size,35)
        self.encoder2=Encoder(input_size,35)
        self.encoder2 = Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.decoder2 = Decoder (35,70,5,output_size)

        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        N = 35  # 사용되는 차원 수
        self.dct_basis = self.create_dct_basis(N)
        self.idct_basis = self.create_idct_basis(N)

    
    def generate(self,z,c):
        z=torch.cat((z,c),dim=-1)
        return self.decoder2(z,c)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def reduce_mean_to_35(self, mean_175):
        # 175차원을 5개의 그룹으로 나누고, 각 그룹의 평균을 구해 35차원으로 축소
        mean_35 = mean_175.view(mean_175.size(0), 5, -1, *mean_175.size()[2:]).mean(dim=1)
        return mean_35
    def create_dct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        dct_basis = torch.sqrt(torch.tensor(2.0/N)) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)
        return dct_basis

    def create_idct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        idct_basis = torch.sqrt(torch.tensor(2.0/N)) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)
        return idct_basis
    
    def apply_dct(self, x):
        return self.dct(x)
    def apply_idct(self, y):
        return self.idct(y)
    

    def dct(self, x):
        # Apply DCT transformation
        return torch.matmul(x, self.dct_basis)
    def idct(self, x):
        # Apply IDCT transformation
        return torch.matmul(x, self.idct_basis)
    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
    
    def p_sample(self, xt,t,c):
        eps_theta,_,out,_,_,_,_,output,_= self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return eps_theta
        #return mean+(var**0.5)*eps
        return mean
         
        #return self.generate(eps_theta,out)
        #return self.generate(eps_theta,c)
        #return mean + (var ** 0.5) * output
        #return output
    def forward(self, x,t,c):
    
        z,mu,logvar=self.encoder(x,c)
        
        #z2,mu2,logvar2 = self.encoder2(x,c)
        #z2,mu2,logvar2 = self.encoder(x,c)
        #z_noise = self.apply_dct(z)
        #noise = torch.randn_like(z)
        #z_noise = self.q_sample(z,t,noise)
        #x_noise= self.q_sample(x,t,noise)
        #z2,mu2,logvar2 = self.encoder2(x_noise,x)
        #noise = torch.randn_like(z)
        #z_noise = self.q_sample(z,t,noise)
        #z_noise = self.apply_idct(z_noise)
        t_emb = self.time_embed(t)
        x1 = self.unet(z,t_emb).to(device="cpu")
        x2 = self.unet2(z,t_emb).to(device="cpu")
         
        #tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,t)
        zc=torch.cat((x1,z),dim=-1)
        zc2=torch.cat((x2,z),dim=-1)
        z_noise2=self.apply_dct(x1)
        z_noise2 = self.q_sample(x1,t,x2)
        z_noise2 = self.apply_idct(z_noise2)
        x3=self.unet(z_noise2,t_emb).to(device="cpu")
        x4=self.unet2(z_noise2,t_emb).to(device="cpu")
        zc3=torch.cat((x3,z),dim=-1)
        zc4 = torch.cat((x4,z),dim=-1)

        return x1,x3,z,mean, var,self.decoder(zc, z),self.decoder2(zc2,z),self.decoder(zc3,z),self.decoder2(zc4,z)
                                                            
        #return self.decoder(zc,z),mean,var
        #return self.decoder(zc,c),mean,var
class DenoiseDiffusion20_Trans(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder2(input_size,35)
        self.encoder2 = Encoder2(input_size,35)
        self.encoder3 = Encoder2(input_size,35)
        #self.encoder4 = Encoder(105,35)
        self.cnn=Conv1dModel()
        self.attn = Attention(dim=35)
        self.decoder=Decoder2(70,35,5,output_size)
        #self.generator =  Generator(input,3*input)
        #self.discriminator= Discriminator(3*input,input)
        #self.mixdecoder =MixedDecoder(35,70,output_size,10,10,10)
        #self.decoder2 = Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,xt2,xt3,t,c,c2,c3):
        eps_theta,_,_= self.forward(xt,xt2,xt3,t,c,c2,c3)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        #return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
        return mean
    def forward(self, x,x2,x3,t,c,c2,c3):
        cc=torch.cat((c,c2),dim=-1)
        cc=torch.cat((cc,c3),dim=-1)
        z,mu,logvar=self.encoder(x,c)
        z2,mu2,logvar2 = self.encoder2(x2,c2)
        z3,mu3,logvar3 = self.encoder3(x3,c3)
        #z=z+z2
        #z=torch.cat((x,x2),dim=-1)
       
        #noise = torch.randn_like(z)
        zt=torch.cat((z,z2),dim=-1)
        zt=torch.cat((zt,z3),dim=-1)
        #z=torch.cat((z,x3),dim=-1)
        tgt=torch.cat((c,c2),dim=-1)
        tgts=torch.cat((tgt,c3),dim=-1)
        #z=self.trans(zt,tgt)
        #z_noise=z
        #z,mu4,logvar4= self.encoder4(z,c)
        #z=z.unsqueeze(-1)
        #z=z.permute(0,2,1) 
        #z=self.cnn(z)
        #z=z.squeeze(1)
        #z=z.unsqueeze(-1)
        #z=z.unsqueeze(-1)
        #z=self.attn(z)
        #z=z.squeeze(-1)
        #z=z.squeeze(-1)
        #t_emb = self.time_embed(t)
        #tgt=torch.cat((tgt,t_emb),dim=-1)
        #t_e = t_emb.repeat(1,2)
        #x= self.trans(z,t_emb)
        #z=self.trans(z,t_e)
        #print(t_emb.shape)
        #t3=t.repeat(1,3)
        
       
        #z_noise = self.q_sample(z,t,noise)
        #z_noise = self.q_sample(z_noise,t,noise)
        #t3_emb=self.time_embed(t3)
        t_emb=self.time_embed(t)
        #z=torch.cat((z,t_emb),dim=-1)
        x = self.trans(z,z2,z3,t_emb,c,tgt,tgts).to(device="cpu")
        # may be ok when using t_emb upper !
        #x = self.unet(x,t_emb).to(device="cpu")
        #x = self.trans(z_noise,t_emb).to(device="cpu")
        #x = self.trans(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        #return x ,mean,var
        return self.decoder(zc,c),mean,var

class DenoiseDiffusion21(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder2(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.encoder2 = Encoder2(input_size,35)
        self.decoder2=Decoder(35,70,5,output_size)
        #self.decoder2 = Decoder(35,70,5,output_size)
        #self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c,c2,c3):
        eps_theta,_,_ = self.forward(xt,t,c,c2,c3)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        #return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
        return mean
    def forward(self, x,t,c,c2,c3):
        #cc=torch.cat((c,c2),dim=-1)
        #cc=torch.cat((cc,c3),dim=-1)
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        #x = self.unet(x,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        #zcc=torch.cat((zc,c3),dim=-1)
        x2=self.decoder(zc,c)
        z2,mu2,logvar2=self.encoder2(x2,c3)
        x2=self.unet(z2,t_emb).to(device="cpu")
        zcc=torch.cat((x2,z2),dim=-1)
        return self.decoder2(zcc,c3),mean,var
        #return self.decoder(zcc,c),mean,var

class DenoiseDiffusion22(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder2(input_size,35)
        self.encoder=Encoder2(input_size,35)
        self.decoder=Decoder(35,105,5,output_size)
        #self.decoder2 = Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=70)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,xt2,t,c,c2,c3):
        eps_theta,_,_ = self.forward(xt,t,c,c2,c3)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        #return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
        return mean
    def forward(self, x,x2,t,c,c2,c3):
        #cc=torch.cat((c,c2),dim=-1)
        #cc=torch.cat((cc,c3),dim=-1)
        z,mu,logvar=self.encoder(x,c)
        z2,mu2,logvar2= self.encoder2(x2,c3)
        tgt=torch.cat((c,c3),dim=-1)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        x2= self.unet(z2,t_emb).to(device="cpu")
        #x = self.unet(x,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        zcc=torch.cat((x2,z2),dim=-1)
        zt=torch.cat((zc,zcc),dim=-1)
        return self.trans(zt,tgt),mean,var
import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LinformerAttention(nn.Module):
    def __init__(self, seq_len, dim, n_heads, k, bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.qw = nn.Linear(dim, dim, bias = bias)
        self.kw = nn.Linear(dim, dim, bias = bias)
        self.vw = nn.Linear(dim, dim, bias = bias)

        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))

        self.ow = nn.Linear(dim, dim, bias = bias)

    def forward(self, x):
        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)

        B, L, D = q.shape
        q = torch.reshape(q, [B, L, self.n_heads, -1])
        q = torch.permute(q, [0, 2, 1, 3])
        k = torch.reshape(k, [B, L, self.n_heads, -1])
        k = torch.permute(k, [0, 2, 3, 1])
        v = torch.reshape(v, [B, L, self.n_heads, -1])
        v = torch.permute(v, [0, 2, 3, 1])
        k = torch.matmul(k, self.E[:L, :])

        v = torch.matmul(v, self.F[:L, :])
        v = torch.permute(v, [0, 1, 3, 2])

        qk = torch.matmul(q, k) * self.scale
        attn = torch.softmax(qk, dim=-1)

        v_attn = torch.matmul(attn, v)
        v_attn = torch.permute(v_attn, [0, 2, 1, 3])
        v_attn = torch.reshape(v_attn, [B, L, D])

        x = self.ow(v_attn)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TransformerBlock(nn.Module):
    def __init__(self, seq_len, dim, heads, mlp_dim, k, rate=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = LinformerAttention(seq_len, dim, heads, k)
        self.ln_2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(rate),
        )
        self.gamma_1 = nn.Linear(dim, dim)
        self.beta_1 = nn.Linear(dim, dim)
        self.gamma_2 = nn.Linear(dim, dim)
        self.beta_2 = nn.Linear(dim, dim)
        self.scale_1 = nn.Linear(dim, dim)
        self.scale_2 = nn.Linear(dim, dim)

        nn.init.zeros_(self.gamma_1.weight)
        nn.init.zeros_(self.beta_1.weight)
        nn.init.zeros_(self.gamma_1.bias)
        nn.init.zeros_(self.beta_1.bias)  

        nn.init.zeros_(self.gamma_2.weight)
        nn.init.zeros_(self.beta_2.weight)
        nn.init.zeros_(self.gamma_2.bias)
        nn.init.zeros_(self.beta_2.bias)  

        nn.init.zeros_(self.scale_1.weight)
        nn.init.zeros_(self.scale_2.weight)
        nn.init.zeros_(self.scale_1.bias)
        nn.init.zeros_(self.scale_2.bias)  

    def forward(self, x, c):
        #c = self.ln_act(c)
        scale_msa = self.gamma_1(c)
        shift_msa = self.beta_1(c)
        scale_mlp = self.gamma_2(c)
        shift_mlp = self.beta_2(c)
        gate_msa = self.scale_1(c).unsqueeze(1)
        gate_mlp = self.scale_2(c).unsqueeze(1)
        x = self.attn(modulate(self.ln_1(x), shift_msa, scale_msa)) * gate_msa + x
        return self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp)) * gate_mlp + x

class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.ln_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
        # Zero-out output layers:
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)        

    def forward(self, x, c):
        scale = self.gamma(c)
        shift = self.beta(c)
        x = modulate(self.ln_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    def __init__(self, img_size, dim=64, patch_size=35,
                 depth=3, heads=7, mlp_dim=512, k=64, in_channels=3):
        super(DiT, self).__init__()
        self.dim = dim
        self.n_patches = (img_size // patch_size)**2 
        self.depth = depth
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches, dim))
        self.patches = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, 
                      stride=patch_size, padding=0, bias=False),
        )
        
        self.transformer = nn.ModuleList()
        for i in range(self.depth):
            self.transformer.append(
                TransformerBlock(
                    self.n_patches, dim, heads, mlp_dim, k)
            )

        self.emb = nn.Sequential(
            TimeEmbedding(70),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.final = FinalLayer(dim, patch_size, in_channels)
        self.ps = nn.PixelShuffle(patch_size)

    def forward(self, x, t):
        t = self.emb(t)
        x=x.unsqueeze(-1)
        x = self.patches(x)
        B, C, H, W = x.shape
        x = x.permute([0, 2, 3, 1]).reshape([B, H * W, C])
        x += self.pos_embedding
        for layer in self.transformer:
            x = layer(x, t)

        x = self.final(x, t).permute([0, 2, 1])
        x = x.reshape([B, -1, H, W])
        x = self.ps(x)
        x=x.squeeze(-1)
        return x
class DenoiseDiffusion23(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet8(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        
        self.trans =TransformerModel(in_channels=105)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,edge,c,c2,c3):
        eps_theta,_,_,_ = self.forward(xt,t,edge,c,c2,c3)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        return mean
        #return mean + (var ** 0.5) * eps
    def forward(self, x,t,edge,c,c2,c3):
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        for i in range(1,self.timesteps):
            x = self.unet(x,t_emb,edge).to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        return self.decoder(zc,c),mean,var


class DenoiseDiffusion24(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.encoder2=Encoder(input_size,35)
        self.encoder3 = Encoder(input_size,35)
        self.encoder4= Encoder(input_size,35)
        self.decoder=Decoder(35,140,5,output_size)
        #self.decoder2 = Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=105)
        self.nn =nn.Linear(140,70)
        self.nn2 = nn.Linear(70,35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        N = 35  # 사용되는 차원 수
        self.dct_basis = self.create_dct_basis(N)
        self.idct_basis = self.create_idct_basis(N)

    def create_dct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        dct_basis = torch.sqrt(torch.tensor(2.0/N)) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)
        return dct_basis

    def create_idct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        idct_basis = torch.sqrt(torch.tensor(2.0/N)) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)
        return idct_basis

    def generate(self,z,c):
        z=torch.cat((z,c),dim=-1)
        return self.decoder(z,c)
    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
    def apply_dct(self, x):
        return self.dct(x)
    def apply_idct(self, y):
        return self.idct(y)
    def p_sample(self, xt,t,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        #return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
        return mean
    def forward(self, x,x2,x3,x4,t,c):
        z,mu,logvar=self.encoder(x,c)
        z2,mu2,logvar2 = self.encoder2(x2,c)
        z3,mu3,logvar3 = self.encoder3(x3,c)
        z4,mu4,logvar4 = self.encoder4(x4,c)
        #zk=torch.cat((z,z2),dim=-1)
        #zk=torch.cat((zk,z3),dim=-1)
        #zk=torch.cat((zk,z4),dim=-1)
        zk=z+z2+z3+z4
        zk=self.apply_dct(zk)
        #zkk=self.nn(zk)
        #zkk=self.nn2(zkk)
        t_emb = self.time_embed(t)
        x = self.unet(zk,t_emb).to(device="cpu")
        x=self.apply_idct(x)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,zk),dim=-1)
        return self.decoder(zc,c),mean,var
import torch
class SEBlock(nn.Module):
    def __init__(self, input_dim, output,reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_dim, input_dim // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(input_dim // reduction, input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: Global Average Pooling
        b, c = x.shape
        y = self.global_avg_pool(x.view(b, c, 1)).view(b, c)
        # Excitation: Fully Connected layers
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        # Scale the input features
        y = y.view(b, c)

        return x * y
class SETransformerBlock(nn.Module):
    def __init__(self, input_dim, output,num_heads, reduction=16):
        super(SETransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim)
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.se_block = SEBlock(input_dim,output, reduction)

    def forward(self, x):
        # Multihead Attention + Residual Connection
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)

        # Feed Forward Network + Residual Connection
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)

        # SEBlock 적용
        x = self.se_block(x)

        return x
import math
import torch.fft
class TransFusion(nn.Module):
    def __init__(self, input_size, noise_steps, latent_size, output_size):
        super().__init__()
        self.timesteps = noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)
        self.encoder = Encoder(input_size, 35)
        self.decoder = Decoder(35, 70, 5, output_size)
        #self.layer = nn.ModuleList([
        #    SETransformerBlock(input_size,output=output_size, num_heads=7, reduction=4)
        #    for _ in range(2)
        #])
        self.down1 = SETransformerBlock(input_size,output=output_size,num_heads=7,reduction=4)
        self.down2 = SETransformerBlock(input_size,output=output_size,num_heads=7,reduction=4)
        self.nn1 =nn.Linear(2*input_size,input_size)
        self.up1 = SETransformerBlock(input_size,output=output_size,num_heads=7,reduction=4)
        self.nn2 = nn.Linear(2*input_size,input_size)
        self.up2 = SETransformerBlock(input_size,output=output_size,num_heads=7,reduction=4)
        self.embedding= nn.Linear(70,35)
        self.trans = TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta = self.gaussiandiffusion.beta
        self.mean = 0
        self.logvar = 0

        # DCT와 IDCT 행렬 미리 생성 후 캐싱
        N = 35  # 사용되는 차원 수
        self.dct_basis = self.create_dct_basis(N)
        self.idct_basis = self.create_idct_basis(N)

    def create_dct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        dct_basis = torch.sqrt(torch.tensor(2.0/N)) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)
        return dct_basis

    def create_idct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        idct_basis = torch.sqrt(torch.tensor(2.0/N)) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)
        return idct_basis
    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
    def apply_dct(self, x):
        return self.dct(x)
    def apply_idct(self, y):
        return self.idct(y)
    def p_sample(self, xt,t,c):
        eps_theta,_,_,_= self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #mean = self.apply_idct(mean)
        #return mean+(var**0.5)
        return mean
        #return mean + (var ** 0.5) * eps
    def dct(self, x):
        return torch.matmul(x, self.dct_basis.to(x.device))

    def idct(self, x):
        return torch.matmul(x, self.idct_basis.to(x.device))
    def forward(self, x, t, c):
        # 인코딩 부분
        z, mu, logvar = self.encoder(x, c)
        z2 = self.apply_dct(x)
        # 시간 임베딩 처리
        t_emb = self.time_embed(t)

        # DCT 적용 + noise 적용
        z_noise = self.apply_dct(z)
        noise = torch.randn_like(z)
        z_noise = self.q_sample(z_noise,t,noise)
        combined_input= torch.cat((z2,z_noise),dim=-1)
        combined_input=self.embedding(combined_input)
        # Transformer Layer 적용
        
        x = self.down1(combined_input)
        xd1= x
        x = self.down2(x)
        xd2 = x
        x=torch.cat((x,xd1),dim=-1)
        x = self.nn1(x)
        x = self.up1(x)
        x = torch.cat((x,xd2),dim=-1)
        x = self.nn2(x)
        x = self.up2(x)

        # Inference용 계산 (p_sample에서 사용됨)
        mean, var = self.q_xt_x0(x, t)

        # Concatenate encoded state
        zc = torch.cat((x, z), dim=-1)

        # 최종 IDCT 적용
        return self.apply_idct(x), mean, var,noise
class DenoiseDiffusion_Trans(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
    def dct(self,x):
        N = x.size(-1)  # Last dimension size
        device = x.device

        # Create DCT basis matrix
        k = torch.arange(N, device=device).view(1, -1)
        n = torch.arange(N, device=device).view(-1, 1)
        dct_basis = torch.sqrt(2 / N) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)  # Normalize the first row

        # Apply DCT basis to input data
        return torch.matmul(x, dct_basis)
    def idct(self,x):
        N = x.size(-1)  # Last dimension size
        device = x.device

        # Create IDCT basis matrix
        k = torch.arange(N, device=device).view(1, -1)
        n = torch.arange(N, device=device).view(-1, 1)
        idct_basis = torch.sqrt(2 / N) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)  # Normalize the first column

        # Apply IDCT basis to input data
        return torch.matmul(x, idct_basis)
    def p_sample(self, xt,t,c):
        eps_theta,_,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean+(var**0.5)
        #return mean
        #return mean + (var ** 0.5) * eps
    def forward(self, x,t,c):
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        z_noise=self.dct(z)
        x=self.trans(z,z_noise)

        #x = self.unet(z,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")

        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        return self.idct(x),mean,var
class DanceEncoder10(nn.Module):
    def __init__(self,pose_size,hidden_size,latent_size):
        super().__init__()
        self.input_size = pose_size*10
        self.pose_size = pose_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size,self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size,self.hidden_size)
        self.mu = nn.Linear(self.hidden_size,self.latent_size)
        self.std = nn.Linear(self.hidden_size,self.latent_size)


    def encode(self,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
        data = torch.cat((t1,t2,t3,t4,t5,t6,t7,t8,t9,t10),dim=1)
        out1 = self.fc1(F.elu(data))
        out2 = self.fc2(F.elu(out1))
        out3 = self.fc3(F.elu(out2))
        out4 = self.fc4(F.elu(out3))
        return self.mu(out4),self.std(out4) 

    def reparameterize(self,mu,var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu+std*eps

    def forward(self,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
        mu , var = self.encode(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10)
        z = self.reparameterize(mu,var)
        return z,mu,var

class DanceDecoder10(nn.Module):
    def __init__(self,latent_size,pose_size,hidden_size,output_size):
        super().__init__()
        self.latent_size = latent_size
        self.pose_size = pose_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.pose_size*5+self.latent_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size+self.pose_size*5,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size+self.pose_size*5,self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size+self.pose_size*5,self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size+self.pose_size*5,self.output_size)

    def forward(self,z,t1,t2,t3,t4,t5):
        out1 = self.fc1(F.elu(torch.cat((z,t1,t2,t3,t4,t5),dim=1)))
        out2 = self.fc2(F.elu(torch.cat((out1,t1,t2,t3,t4,t5),dim=1)))
        out3 = self.fc3(F.elu(torch.cat((out2,t1,t2,t3,t4,t5),dim=1)))
        out4 = self.fc4(F.elu(torch.cat((out3,t1,t2,t3,t4,t5),dim=1)))
        return self.fc5(torch.cat((out4,t1,t2,t3,t4,t5),dim=1))

class DanceVAE10(nn.Module):
    def __init__(self,pose_size,encode_hidden_size,latent_size,decode_hidden_size,output_size):
        super().__init__()
        self.encoder = DanceEncoder10(pose_size,encode_hidden_size,latent_size)
        self.decoder = DanceDecoder10(latent_size,pose_size,decode_hidden_size,output_size)
        self.pose_data_mu = 0
        self.pose_data_std = 0
    
    def encode(self,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
        z,mu,logvar = self.encoder(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10)
        return z,mu,logvar

    def forward(self,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
        z,mu,logvar= self.encoder(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10)
        return self.decoder(z,t1,t3,t5,t7,t9),mu,logvar

    def sample(self,z,t1,t3,t5,t7,t9):
        return self.decoder(z,t1,t3,t5,t7,t9)

    def set_normalize(self,pose_mu,pose_std):
        self.pose_data_mu = pose_mu
        self.pose_data_std = pose_std

    def normalize_pose(self,x):
        return (x-self.pose_data_mu)/self.pose_data_std
    
    def denormalize_pose(self,x):
        return x*self.pose_data_std+self.pose_data_mu

class TrackerEncoder(nn.Module):
    def __init__(self,tracker_size,hidden_size,latent_size):
        super().__init__()
        self.input_size = tracker_size*10
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size+self.input_size,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size+self.input_size,self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size+self.input_size,self.hidden_size)
        self.mu = nn.Linear(self.hidden_size+self.input_size,self.latent_size)
        self.std = nn.Linear(self.hidden_size+self.input_size,self.latent_size)

    def encode(self,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
        data = torch.cat((t1,t2,t3,t4,t5,t6,t7,t8,t9,t10),dim=1)
        out1 = self.fc1(F.elu(data))
        out2 = self.fc2(F.elu(torch.cat((out1,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10),dim=1)))
        out3 = self.fc3(F.elu(torch.cat((out2,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10),dim=1)))
        out4 = self.fc4(F.elu(torch.cat((out3,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10),dim=1)))
        return self.mu(torch.cat((out4,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10),dim=1)),self.std(torch.cat((out4,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10),dim=1)) 

    def reparameterize(self,mu,var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu+std*eps

    def forward(self,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
        mu , var = self.encode(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10)
        z = self.reparameterize(mu,var)
        return z,mu,var

class TrackerDecoder(nn.Module):
    def __init__(self,latent_size,tracker_size,hidden_size,output_size):
        super().__init__()
        self.tracker_size = tracker_size
        self.input_size = latent_size+self.tracker_size*5
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size+latent_size,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size+latent_size,self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size+latent_size,self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size+latent_size,self.output_size)

    def forward(self,z,t1,t3,t5,t7,t9):
        out1 = self.fc1(F.elu(torch.cat((z,t1,t3,t5,t7,t9),dim=1)))
        out2 = self.fc2(F.elu(torch.cat((out1,z),dim=1)))
        out3 = self.fc3(F.elu(torch.cat((out2,z),dim=1)))
        out4 = self.fc4(F.elu(torch.cat((out3,z),dim=1)))
        return self.fc5(torch.cat((out4,z),dim=1))

class TrackerVAE(nn.Module):
    def __init__(self,tracker_size,encode_hidden_size,latent_size,decode_hidden_size,output_size):
        super().__init__()
        self.encoder = TrackerEncoder(tracker_size,encode_hidden_size,latent_size)
        self.decoder = TrackerDecoder(latent_size,tracker_size,decode_hidden_size,output_size)
    
    def encode(self,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
        z,mu,logvar = self.encoder(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10)
        return z,mu,logvar

    def forward(self,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
        z,mu,logvar= self.encoder(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10)
        return self.decoder(z,t1,t3,t5,t7,t9),mu,logvar

    def sample(self,z,t1,t3,t5,t7,t9):
        return self.decoder(z,t1,t3,t5,t7,t9)


class TrackerAutoEncoder(nn.Module):
    def __init__(self,tracker_size,num_condition_frames,hidden_size,latent_size):
        super().__init__()
        self.input_size = tracker_size*num_condition_frames
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.fc1 = nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size+self.input_size,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size+self.input_size,self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size+self.input_size,self.latent_size)

    def encode(self,tracker_data):
        data = tracker_data.flatten(-2)
        out = F.elu(self.fc1(data))
        out = F.elu(self.fc2(torch.cat((out,data),dim=1)))
        out = F.elu(self.fc3(torch.cat((out,data),dim=1)))
        out = self.fc4((torch.cat((out,data),dim=1)))
        return out

    def forward(self,tracker_data):
        latent = self.encode(tracker_data)
        return latent 

class TrackerAutoDecoder(nn.Module):
    def __init__(self,latent_size,tracker_size,num_condition_frames,hidden_size,output_size):
        super().__init__()
        self.input_size = latent_size
        self.tracker_size = tracker_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size+self.input_size,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size+self.input_size,self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size+self.input_size,self.output_size)

    def forward(self,latent,tracker):
        out = F.elu(self.fc1(latent))
        out = F.elu(self.fc2(torch.cat((out,latent),dim=1)))
        out = F.elu(self.fc3(torch.cat((out,latent),dim=1)))
        out = self.fc4(torch.cat((out,latent),dim=1))
        return out

class TrackerAuto(nn.Module):
    def __init__(self,tracker_size,num_condition_frames,encoder_hidden_size,latent_size,decoder_hidden_size,output_size):
        super().__init__()
        self.encoder = TrackerAutoEncoder(tracker_size,num_condition_frames,encoder_hidden_size,latent_size)
        self.decoder = TrackerAutoDecoder(latent_size,tracker_size,num_condition_frames,decoder_hidden_size,output_size)
        self.num_condition_frames = num_condition_frames
        #self.decoder = MixedDecoder(35,latent_size,decoder_hidden_size,0,1,2)

    def forward(self,tracker_data):
        z = self.encoder(tracker_data)
        return self.decoder(z,tracker_data[:,int(self.num_condition_frames/2-1),:])
    
class CNN(nn.Module):
    def __init__(self,tracker_size,condition_size,output_size):
        super().__init__()
        self.tracker_size = tracker_size*condition_size
        self.output_size = output_size
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,512,(1,3),stride=(1,3)),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512,32,1),
            torch.nn.ELU(),
        )
        self.fc1 = torch.nn.Linear(19200,1024)
        self.fc2 = torch.nn.Linear(1024,output_size)


    def forward(self,history):
        history = history.unsqueeze(1)
        #print(history.shape)
        out = self.layer1(history)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = out.view(out.shape[0],-1)
        #print(out.shape)
        out = F.elu(self.fc1(out))
        out = self.fc2(out)
        #print(out.shape)
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EncoderOnly(nn.Module):
    def __init__(self, input_size, condition_size, latent_size, hidden_size=512):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        real_input = self.input_size + condition_size
        self.fc1 = nn.Linear(real_input, self.hidden_size)
        self.fc2 = nn.Linear(input_size + self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(input_size + self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(input_size + self.hidden_size, self.hidden_size)
        self.fc5 = nn.Linear(input_size + self.hidden_size, latent_size)

    def encode(self, input, condition_input):
        out1 = F.elu(self.fc1(torch.cat((input, condition_input), dim=1)))
        out2 = F.elu(self.fc2(torch.cat((input, out1), dim=1)))
        out3 = F.elu(self.fc3(torch.cat((input, out2), dim=1)))
        out4 = F.elu(self.fc4(torch.cat((input, out3), dim=1)))
        out5 = torch.cat((input, out4), dim=1)
        return self.fc5(out5)

    def forward(self, input, condition_input):
        output = self.encode(input, condition_input)
        return output


class VectorQuantizer2(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer2, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, z):
        z = z.view(-1, self.embedding_dim)
        distances = (torch.sum(z ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(z, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        z_q = torch.index_select(self.embeddings.weight, dim=0, index=encoding_indices.squeeze()).view_as(z)

        loss = F.mse_loss(z_q, z.detach()) + self.commitment_cost * F.mse_loss(z_q.detach(), z)

        z_q = z + (z_q - z).detach()

        return z_q, loss


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
               torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class MoEDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, hidden_dim):
        super(MoEDecoder, self).__init__()
        self.num_experts = num_experts

        # Experts: 각 전문가에 대해 별도의 네트워크를 생성
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        ) for _ in range(num_experts)])

        # Gating network: 각 전문가의 가중치를 결정
        self.gating_network = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Gating network를 통해 각 전문가의 가중치 계산
        gate_values = F.softmax(self.gating_network(x), dim=-1)

        # 각 전문가의 출력을 계산하고, 가중치를 곱한 후 합산
        output = sum(gate_values[:, i].unsqueeze(1) * self.experts[i](x) for i in range(self.num_experts))

        return output


class VQVAE(nn.Module):
    def __init__(self, input_size, hidden_dim, latent_size, output_size,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False, input_frame=1, expert_num=8,
                 input_mult_frame=False, output_mult_frame=False):

        super(VQVAE, self).__init__()
        self.encoder = EncoderOnly(input_size * input_frame, input_size, latent_size, hidden_dim)

        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer2(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = MoEDecoder(embedding_dim + input_size, latent_size, expert_num, hidden_dim)

        if output_mult_frame == False:
            self.decoder2 = MoEDecoder(latent_size + input_size, output_size, expert_num, hidden_dim)
        else:
            self.decoder2 = MoEDecoder(latent_size + input_size, output_size * input_frame, expert_num, hidden_dim)

        self.data_std = 0
        self.data_avg = 0
        self.config = {'model_name': 'VQVAE',
                       'input': input_size,
                       'latent': latent_size,
                       'output_size': output_size,
                       'embeddings': n_embeddings,
                       'embedding_dim': embedding_dim,
                       'beta': beta,
                       'input_frame': input_frame,
                       'isRecursive': False
                       }

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def get_param(self):
        return self.config

    def set_normalization(self, std, avg):
        self.data_std = std
        self.data_avg = avg

    def normalize(self, t):
        return (t - self.data_avg) / self.data_std

    def denormalize(self, t):

        return t * self.data_std + self.data_avg

    def forward(self, x, condition2, verbose=False):

        z_e = self.encoder(x, condition2)
        # pdb.set_trace()
        # z_e = self.pre_quantization_conv(z_e)
        z_q, loss = self.vector_quantization(
            z_e)
        # x_hat = self.decoder(z_q, condition2)
        x_hat = self.decoder(torch.cat((z_q, condition2), dim=1))
        x_hat = self.decoder2(torch.cat((x_hat, condition2), dim=1))

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return loss, x_hat

