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

class Encoder(nn.Module):
    def __init__(self,input_size,latent_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = 256
        real_input = self.input_size*2
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

class Decoder(nn.Module):
    def __init__(self,input_size,latent_size,num_experts,output_size):
        super().__init__()
        input_size = latent_size + input_size
        output_size = output_size
        hidden_size = 256
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size+latent_size,hidden_size)
        self.fc3 = nn.Linear(latent_size+hidden_size,output_size)

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


class BetaDerivatives():
    def __init__(self,time_steps,beta_start,beta_end):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.time_steps = time_steps
        self.betas = self.prepare_noise_schedule().to(device="cpu")
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
        emb = torch.log(torch.tensor(1000.0)/(half_dim-1))
        emb = torch.exp(torch.arange(half_dim)*-emb)
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
        #x=self.n0(x)


        #print(x.shape)
        x=x+t



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

# Denoising Diffusion Model
class DenoiseDiffusion(nn.Module):
    def __init__(self,input_size,output_size,noise_steps):
        super().__init__()
        #self.input_size = input_size
        self.input_size= input_size
        self.output_size = output_size
        self.noise_steps = noise_steps
        self.hidden_size = 1024
        self.time_dim=70
        self.time_dim2=self.time_dim
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
        self.fc6 = nn.Linear(self.hidden_size + self.time_dim2, self.hidden_size)
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
        return mean+(var**0.5)*eps

    def p_sample(self,xt,t):
    #def p_sample(self,xt,t):
        #eps_theta=self.forward(xt,t)
        eps_theta = self.forward(xt,t)
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[t]
        alpha = self.gaussiandiffusion.alpha[t]

        eps_coef = (1-alpha)/(1-alpha_hat)**0.5
        #eps_coef2=eps_coef.repeat(1,5)
        alpha2=alpha.repeat(1,5)
        mean = 1/(alpha2**0.5)*(xt-eps_coef*eps_theta)
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
        eps_theta = self.forward(xt,t)
        tt=torch.tensor(t).long().to(device="cpu")
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
        return x

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
        tt=torch.tensor(10).long()
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

class DenoiseDiffusion7(nn.Module):
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




