import copy
from json import load
import json
from types import SimpleNamespace
import time 
from multiprocessing import Condition
import os
from random import randint, random
import sys
import platform
import pdb
from unittest import loader
from xml.dom import minicompat # use pdb.set_trace() for debugging
sys.path.append(os.getcwd())
import libmainlib as m   
import luamodule as lua  # see luamodule.py
import numpy as np 
import torch 
from gym_mp.models2 import Encoder,Decoder,VAE, VQVAE,DenoiseDiffusion14_mul,Belfusion,DenoiseDiffusion14_mul_vec,DenoiseDiffusion14_mul_vec_n_atten
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler , RandomSampler
from tensorboardX import SummaryWriter 
import torch.nn.functional as F 
import settings
# simply forward UI events to lua
def onCallback(mid, userdata):
    lua.onCallback(mid, userdata)

def onFrameChanged(currFrame):
    lua.onFrameChanged(currFrame)

def frameMove(fElapsedTime):
    lua.frameMove(fElapsedTime)

def handleRendererEvent(ev, button, x,y):
    return lua.handleRendererEvent(ev, button, x,y)

def lua_getdim():
    l=m.getPythonWin()
    l.getglobal("get_dim")
    l.call(0,1)
    info= l.popvectorn()
    state_dim=int(info.get(0))
    action_dim=int(info.get(1))
    # print(state_dim)
    # print(action_dim)
    return state_dim,action_dim

def lua_getIframe(iframe):
    l=m.getPythonWin()
    l.getglobal("seok_getIframeDOF")
    input = m.vectorn()
    input.setSize(0)
    input = iframe
    # print(input)
    l.push(input)
    l.call(1,1)
    return l.popvectorn()

def lua_getFrame():
    l = m.getPythonWin()
    l.getglobal("get_frame")
    l.call(0,1)
    return l.popnumber()

def Tonumpy(data):
    return np.array(data.ref())

def Discontinuity():
    l = m.getPythonWin()
    l.getglobal("getDiscontinuity")
    l.call(0,1)
    return l.popvectorn()

def Alldata_load():
    l=m.getPythonWin()
    l.getglobal("All_data_frames")
    l.call(0,1)
    return l.popmatrixn()

def main():
    
    with open("config/dif_multi_n_attn.json", "r") as file:
        config = json.load(file)
    
    train = True
    
    load_save_model = False
    latent_size = config["train_parameter"]["latent_size"]
    #walk : 64~128 stitch:512~1024
    mini_batch = config["train_parameter"]["mini_batch"]
    teacher_epochs = config["train_parameter"]["teacher_epochs"]
    ramping_epochs = config["train_parameter"]["ramping_epochs"]
    student_epochs = config["train_parameter"]["student_epochs"]
    num_epochs = teacher_epochs+ramping_epochs+student_epochs
    num_experts = config["train_parameter"]["num_experts"]
    input_frames = config["model"]["input_frames"]
    initial_lr = config["train_parameter"]["initial_lr"]
    # final_lr = 1e-7
    prediction_frames = config["train_parameter"]["recursive_frames"]
    condition_frame = config["train_parameter"]["condition_frames"]
    hidden_dim = config["train_parameter"]["hidden_dim"]
    codebook_size = config["train_parameter"]["code_dim"]
    beta = config["train_parameter"]["beta"]
    pt_path = config["path"]["pt_path"] + config["model"]["pt_name"] + ".pt"
    timestep = config["train_parameter"]["time_steps"]
    option=''
    if len(sys.argv)==1:
        scriptFile = config["path"]["lua_path"]
    elif len(sys.argv)==2:
        scriptFile=config["path"]["lua_path"]
        if sys.argv[1] == 'False':
            train = False
            settings.train = False
        else :
            train = True
            settings.train = True
    elif len(sys.argv)==3:
        option=sys.argv[1]
        scriptFile=sys.argv[2] 
    uiscale=1.5
    
    if train :
        device =torch.device("cpu")
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    if platform.system()=='Darwin':
        m.createMainWin(int((600+220)*uiscale),int((400+100)*uiscale), int(600*uiscale), int(400*uiscale),uiscale, "../Resource/ogreconfig_mac.txt", "plugins_mac.cfg", "ogre_mac.cfg")
    else:
        if option=='--sep':
           m.createMainWin(int((10+220)*uiscale),int((400+100)*uiscale), int(10*uiscale), int(400*uiscale),uiscale, "../Resource/ogreconfig_linux_sepwin.txt", "plugins_linux.cfg", "ogre_linux.cfg")
        else:
            m.createMainWin(int((1024+180)*uiscale),int((600+100)*uiscale), int(1024*uiscale), int(600*uiscale),uiscale)
    m.showMainWin()
    if scriptFile[0:1]!='.':
        scriptFile=os.path.relpath(scriptFile, os.getcwd())
    if scriptFile[-3:]!='lua':
        scriptFile=scriptFile+'.lua'
    l=m.getPythonWin()

    print('loading', scriptFile)
    #loadScrit == ctor 까지 수행됨
    l.loadScript(scriptFile)
    # mode_setting function
    # python에서 argument, return 받아오는 법
    # getglobal ("함수 이름")
    l.getglobal('mode_setting')
    # push (넣으려고 하는 parameter "스택"형식으로 쌓기)
    l.push(settings.train)
    # call로 불러 오면서 return (앞이 인자개수, 리턴 개수)
    l.call(1,0)
    # 써진 반대로 (스택이니까) pop! 

    mocap_data = torch.from_numpy(Tonumpy(Alldata_load())).float().to(device)
    if torch.isnan(mocap_data).any():
        print("mocap data has nan value")

    discont = Tonumpy(Discontinuity())
    discont = list(map(int,discont))
    mocap_size = mocap_data.shape[0] #numframes = 26909
    frame_size = mocap_data.shape[1] #frame_size = 35
    input_size = frame_size
    output_size = frame_size


    ######data normalize##########
    avg = mocap_data.mean(dim=0)
    std = mocap_data.std(dim=0)
    for i in range(len(std)):
        if std[i] == 0:
            std[i] = 1
    mocap_data = (mocap_data-avg)/std
    ##############################
    #vqvae = VQVAE(input_size,hidden_dim, latent_size, output_size, codebook_size, latent_size, beta, False, input_frames, num_experts, True, False).to(device)
    #vqvae.set_normalization(std.to(device="cpu"),avg.to(device="cpu"))
    vqvae = DenoiseDiffusion14_mul_vec_n_atten(input_size,timestep,latent_size,output_size).to(device)
    if load_save_model :
        print("loading model")
        vqvae = torch.load("vqvae_model_10_large_to_one.pt",map_location=device)

    vqvae.train()
    vqvae_optimizer = optim.Adam(vqvae.parameters(),lr=initial_lr)
    scheduler_lr = optim.lr_scheduler.StepLR(vqvae_optimizer,step_size=10,gamma=0.9)
    writer = SummaryWriter(comment="MVQVAE"+"\nenv_name" + scriptFile + "\nbatch size"+str(num_epochs) + "\nmini batch size" + str(mini_batch) + "\nlearning rate"+str(vqvae_optimizer.param_groups[0]['lr']))
    all_indices = np.linspace(0,mocap_size-1,mocap_size)
    all_indices = np.linspace(0,mocap_size-1,mocap_size)
    bad_indices = []
    for i in range(len(discont)-1):
        for j in range(12):
            bad_indices.append(discont[i+1] - j)
    bad_indices.sort()
    good_mask = np.isin(all_indices,bad_indices,assume_unique=True,invert=True)
    selectable_indices = all_indices[good_mask]
    sample_schedule = torch.cat(
    (
        # First part is pure teacher forcing
        torch.zeros(teacher_epochs),
        # Second part with schedule sampling
        torch.linspace(0.0,1.0,ramping_epochs),
        # last part is pure student
        torch.ones(student_epochs),
    ))
    
    print(selectable_indices)
    if train:
        shape = (mini_batch,condition_frame,output_size)
        input_shape = (mini_batch,input_frames,output_size)
        history = torch.empty(shape).to(device)
        encode_buffer = torch.empty(input_shape).to(device)
        
        start = time.time()
        s1=time.time()
        for ep in range(1,num_epochs+1):
            sampler = BatchSampler(SubsetRandomSampler(selectable_indices),mini_batch,drop_last=True)
            ep_recon_loss = 0
            ep_q_loss = 0
            ep_noise_loss =0
            ep_noise_loss2 =0
            ep_latent_loss =0
            ep_recon_loss2 =0
            ep_ik_loss=0
            ep_ik_loss2=0
            ep_ik_loss3=0
            ep_ground_vector_loss =0
            ep_policy_loss=0
            num_of_minibatch = 1  
            
            for num_of_minibatch,indices in enumerate(sampler):
                t_indices = torch.LongTensor(indices)

                t_indices += 1 
                condition_range = (
                t_indices.repeat((1, 1)).t()
                + torch.arange(0, -1, -1).long()
                )
                history[:,:1].copy_(mocap_data[condition_range])
                in_range =condition_range
                
                
                for i in range(1, input_frames):
                    in_range = torch.cat((in_range[:,0].unsqueeze(1)-1,in_range),dim=1)
                    
                encode_buffer = mocap_data[in_range]
                
                
                for offset in range(1,prediction_frames):
                    use_student = torch.rand(1) < sample_schedule[ep - 1]
                    prediction_range = (
                            t_indices.repeat((1, 1)).t()
                            + torch.arange(offset, offset + 1).long()
                        )
                    t = torch.randint(0, timestep, (mini_batch, input_size,), device=device).long()
                    ground_truth = mocap_data[prediction_range]
                    condition = history[:, :1]
                    condition = condition.flatten(start_dim=1, end_dim=2)
                    ground_truth = ground_truth.flatten(start_dim=1, end_dim=2)
                    #curr_pose=mocap_data[t_indices]
                    curr_pose = mocap_data[t_indices]
                    curr_pose0 = encode_buffer[:,0,:]
                    curr_pose1 = encode_buffer[:,1,:]
                    curr_pose2 = encode_buffer[:,2,:]
                    curr_pose5 = encode_buffer[:,5,:]
                    curr_pose9 = encode_buffer[:,9,:]
                    curr_pose10 = encode_buffer[:,10,:]
                    curr_vec1=curr_pose1-curr_pose0
                    curr_vec2=curr_pose2-curr_pose0
                    curr_vec3=curr_pose9-curr_pose
                    #future_pose = mocap_data[t_indices+10]
                    #curr_pose_vec = (mocap_data[t_indices]-mocap_data[t_indices-1])*0.01
                    #curr_pose_vec = mocap_data[t_indices+prediction_frames-1]-mocap_data[t_indices]
                    curr_pose_vec = ground_truth - curr_pose
                    #curr_pose_vec = future_pose-curr_pose
                    #future_pose = mocap_data[(prediction_frames+t_indices)%(mocap_size)]
                    # t = torch.tensor(condition).long().to(device="cpu")
                    t_check=time.time()
                    t_check2=t_check-s1
                    time_tensor=torch.full((2048,35),0)
                    s1=t_check
                    noise = torch.randn_like(curr_pose_vec)
                    curr_pose_noise = vqvae.q_sample(curr_pose_vec, t, noise)
                    curr_pose_nn = vqvae.q_sample(curr_pose,t,noise)
                    ground_noise = vqvae.q_sample(ground_truth,t,noise)
                    #future_pose_noise = vqvae.q_sample(future_pose,t,noise)
                    # curr_pose_size= vae.sample(input_size)
                    # print(curr_pose_size.shape)
                    # noise_vec=ground_noise-curr_pose_noise......
                    condition_noise = vqvae.q_sample(condition, t) #noise policy in this code
                    #curr_pose_n = vqvae.p_sample(curr_pose, t, condition)# strongly comming in no policy no falliure inside this code!
                    #output, mu, logvar = vqvae(curr_pose_noise, t, condition)
                    r=0.99
                    r2=r*r
                    curr_flow=torch.cat((curr_pose_noise,curr_pose),dim=-1)
                    curr_flow_box=torch.cat((curr_flow,noise),dim=-1)
                    #output, mu , logvar =vqvae(curr_pose_noise,t,condition)
                    output,mu,logvar,ik1,ik_mid,ik2,n_oise,ik_ground_vec = vqvae(curr_pose_nn,curr_pose_noise,curr_vec1,curr_vec2,t,condition,curr_pose,curr_pose_vec)
                    
                    curr_pose_check=vqvae.q_sample(curr_pose,t,noise)

                    curr_pose_vec_check=vqvae.q_sample(curr_pose_vec,t,n_oise)
                    curr_pose_n = vqvae.p_sample(curr_pose_check,curr_pose_vec_check,curr_vec1,curr_vec2, t, condition,output,ik_ground_vec) # In this part we using the reinforcement learning that is environment of frame and noise.....
                    

                    
                    #future_frame = vqvae.p_sample(curr_pose,t,condition)
                    #ouput2,mu2,logvar2 = vqvae(future_pose_noise,t,condition)
                    # curr_pose_nn=curr_pose_n[99]
                    # print(ground_truth.shape)
                    #ground_truth = mocap_data[prediction_range]
                    #conditionfr = encode_buffer[:,input_frames-1]
                   
                    #ground_truth = ground_truth.flatten(start_dim=1,end_dim=2)
                    #curr_pose = encode_buffer.flatten(start_dim=1, end_dim=2)
                    #loss, x_hat = vqvae(curr_pose,conditionfr)
                    
                    history = history.roll(1,dims=1)
                    next_frame = output if use_student else ground_truth
                    #encode_buffer = encode_buffer.roll(-1, dims=1)
                    #encode_buffer[:,input_frames-1].copy_(next_frame.detach())
                    ik_cur=curr_pose0-ik1
                    ik_ground=curr_pose9-ik2
                    
                    history[:,0].copy_(next_frame.detach())
                    recon_loss = (n_oise - noise).pow(2).mean(dim=(0,-1))
                    recon_loss = recon_loss.sum()
                    #recon_loss2 = (ground_truth-Pose).pow(2).mean(dim=(0,-1))
                    #recon_loss2 =(future_frame-ground_truth).pow(2).mean(dim=(0,-1))
                    #recon_loss2 = recon_loss2.sum()
                    #good of policy network
                    recon_loss2 = (curr_pose_n-curr_pose10).pow(2).mean(dim=(0,-1))
                    recon_loss2 = recon_loss2.sum()
                    
                    recon_loss3_1 = (curr_pose0-ik1).pow(2).mean(dim=(0,-1))
                    recon_loss3_2 =(curr_pose9-ik2).pow(2).mean(dim=(0,-1))
                    recon_loss3_3 =(curr_pose5-ik_mid).pow(2).mean(dim=(0,-1))
                    #over 3 is ik loss in this code
                    recon_loss3_1 = recon_loss3_1.sum()
                    recon_loss3_2 = recon_loss3_2.sum()
                    recon_loss3_3 = recon_loss3_3.sum()
                    loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum().clamp(max=0)
                    loss /= logvar.numel()
                    vqvae_optimizer.zero_grad()
                    recon_loss4 = (ground_truth-output).pow(2).mean(dim=(0,-1))
                    recon_loss4 = recon_loss4.sum() 
  
                    recon_loss5 = (ik_ground_vec-curr_pose_vec).pow(2).mean(dim=(0,-1))
                    recon_loss5 = recon_loss5.sum()
                    
                    
                    #(recon_loss+loss).backward()
                    (loss+recon_loss+recon_loss2+r2*recon_loss3_1+r*recon_loss3_2+recon_loss3_3+recon_loss4+recon_loss5).backward()
                    #(recon_loss2).backward()

                    vqvae_optimizer.step()
                    
                    ep_q_loss += float(loss)/prediction_frames
                    ep_recon_loss += float(recon_loss4) / prediction_frames
                    ep_recon_loss2 += float(recon_loss2) / prediction_frames
                    ep_ik_loss += float(recon_loss3_1)/prediction_frames
                    ep_ik_loss2 += float(recon_loss3_2)/prediction_frames
                    ep_ik_loss3 += float(recon_loss3_3)/prediction_frames
                    ep_latent_loss+= float(recon_loss)/prediction_frames
                    ep_ground_vector_loss += float(recon_loss5)/prediction_frames
                    
                   
                  
            avg_recon_loss = ep_recon_loss / mini_batch
            avg_kl_loss = ep_q_loss / mini_batch
            avg_recon_loss2 = ep_recon_loss2/mini_batch
            avg_ik_recon = ep_ik_loss/mini_batch
            avg_ik_recon2 = ep_ik_loss2/mini_batch
            avg_ik_recon3 = ep_ik_loss3/mini_batch
            avg_latent_loss =ep_latent_loss/mini_batch
            avg_ground_vec_loss =ep_ground_vector_loss/mini_batch
            scheduler_lr.step()
            end = time.time()
            print("epoch : {ep}, ep_noise_loss : {ep_recon_loss:0.08f},q:{q:0.08f} p:{p:0.08f} ik_loss:{ik:0.08f},ik_loss2:{ik2:0.08f},ik_loss3:{ik3:0.08f},latent:{lat:0.08f} vec:{vec:0.08f}   learning_rate : {lr:0.07f} , FPS : {FPS}".format(ep=ep,ep_recon_loss=avg_recon_loss,q=avg_kl_loss,p=avg_recon_loss2,ik=avg_ik_recon,ik2=avg_ik_recon2,ik3=avg_ik_recon3,lat=avg_latent_loss,vec=avg_ground_vec_loss,lr=vqvae_optimizer.param_groups[0]['lr'],FPS=int((ep/(end-start))*100)))
            writer.add_scalar('ep_noise_loss',avg_recon_loss,ep)
            writer.add_scalar('ep_kl_loss',avg_kl_loss,ep)
            writer.add_scalar('ep_recon2_loss',avg_recon_loss,ep)
            writer.add_scalar('ep_policy_loss',avg_recon_loss2,ep)
            writer.add_scalar('ep_ik_lsoss',avg_ik_recon,ep)
            writer.add_scalar('ep_ik_loss2',avg_ik_recon2,ep)
            writer.add_scalar('ep_ik_loss3',avg_ik_recon3,ep)
            writer.add_scalar('ep_latetn_loss',avg_latent_loss,ep)
            writer.add_scalar('ep_ground_vec_loss',avg_ground_vec_loss,ep)
            writer.add_scalar('learning_rate',vqvae_optimizer.param_groups[0]['lr'],ep)
    
           
           

          
            torch.save(copy.deepcopy(vqvae).cpu(), pt_path)
    else:
        settings.DIF1016 = torch.load(pt_path)
        m.startMainLoop()
        
def getTrain(flag):
    if settings.train == True:
        tmp = np.array([1])
        flag.ref()[:] = tmp
    else:
        tmp = np.array([0])
        flag.ref()[:] = tmp
        
        
    return flag


    
    
def test_dif16(latent,condition_l,latent2,vae_output):
    with torch.no_grad():
        settings.DIF1016.eval()
        latent_vector = torch.tensor(Tonumpy(latent)).float()
        condition = torch.tensor(Tonumpy(condition_l)).float()
        #condition_1 = torch.tensor(Tonumpy(condition_l)).long()
        #condition = settings.DIF10.q_sample(condition,0)
        latent_vector2=torch.tensor(Tonumpy(latent2)).long()
        latent_vector2=latent_vector2.view(1,-1)
        latent_vector=latent_vector.view(1,-1)
        condition=condition.view(1,-1)
        #print(condition.shape)
        #output=settings.DIF1016.q_sample(condition,1)
        #output = settings.DIF1010cur.p_sample(condition,latent_vector)
        #output=settings.DIF1015ori.p_sample(output,latent_vector2,condition)
        output = settings.DIF1016.p_sample(condition, latent_vector2, condition)
        output = output.detach().numpy()
        vae_output.ref()[:] = output
        #print(vae_output.shape)
        return vae_output
def test_dif316(latent,latent2,latent3,latent4,latent5,latent6,latent7,latent8,latent9,condition_l,latent0,vae_output):
    with torch.no_grad():
        settings.DIF1016.eval()
        latent_vector = torch.tensor(Tonumpy(latent)).float()
        latent_vector2 = torch.tensor(Tonumpy(latent2)).float()
        latent_vector3 = torch.tensor(Tonumpy(latent3)).float()
        latent_vector4 = torch.tensor(Tonumpy(latent4)).float()
        latent_vector5 = torch.tensor(Tonumpy(latent5)).float()
        latent_vector6 = torch.tensor(Tonumpy(latent6)).float()
        latent_vector7 = torch.tensor(Tonumpy(latent7)).float()
        latent_vector8 = torch.tensor(Tonumpy(latent8)).float()
        latent_vector9 = torch.tensor(Tonumpy(latent9)).float()
        
        condition = torch.tensor(Tonumpy(condition_l)).float()
    
        #condition_1 = torch.tensor(Tonumpy(condition_l)).long()
        #condition = settings.DIF10.q_sample(condition,0)
        latent_vector0=torch.tensor(Tonumpy(latent0)).long()
        latent_vector0=latent_vector0.view(1,-1)
        latent_vector=latent_vector.view(1,-1)
        latent_vector2=latent_vector2.view(1,-1)
        latent_vector3=latent_vector3.view(1,-1)
        latent_vector4=latent_vector4.view(1,-1)
        latent_vector5=latent_vector5.view(1,-1)
        latent_vector6=latent_vector6.view(1,-1)
        latent_vector7=latent_vector7.view(1,-1)
        latent_vector8=latent_vector8.view(1,-1)
        latent_vector9=latent_vector9.view(1,-1)
       
        condition=condition.view(1,-1)
        curr_vec=latent_vector9-latent_vector8
        curr_vec1=latent_vector8-latent_vector7
        curr_vec2=latent_vector9-latent_vector7
        #print(condition.shape)
        #output=settings.DIF1016.q_sample(condition,1)
        #output = settings.DIF1010cur.p_sample(condition,latent_vector)
        #output=settings.DIF1015ori.p_sample(output,latent_vector2,condition)
        #output= settings.DIF1016.p_sample(condition_vec,latent_vector0,latent_vector)
        output = settings.DIF1016.p_sample(latent_vector,curr_vec,condition,curr_vec1,curr_vec2,latent_vector0, latent_vector9)     
        #output = settings.DIF1016.p_sample(output, latent_vector0, latent_vector5)
        #output = settings.DIF1016.p_sample(output,latent_vector0,condition)
        #output =settings.DIF1016.generate(output,condition)
        output = output.detach().numpy()
        vae_output.ref()[:] = output
        #print(vae_output.shape)
        return vae_output
def test_dif516(latent,latent2,latent3,latent4,latent5,latent6,latent7,latent8,latent9,condition_l,latent0,vae_output):
    with torch.no_grad():
        settings.DIF1016.eval()
        latent_vector = torch.tensor(Tonumpy(latent)).float()
        latent_vector2 = torch.tensor(Tonumpy(latent2)).float()
        latent_vector3 = torch.tensor(Tonumpy(latent3)).float()
        latent_vector4 = torch.tensor(Tonumpy(latent4)).float()
        latent_vector5 = torch.tensor(Tonumpy(latent5)).float()
        latent_vector6 = torch.tensor(Tonumpy(latent6)).float()
        latent_vector7 = torch.tensor(Tonumpy(latent7)).float()
        latent_vector8 = torch.tensor(Tonumpy(latent8)).float()
        latent_vector9 = torch.tensor(Tonumpy(latent9)).float()
        
        condition = torch.tensor(Tonumpy(condition_l)).float()
        #condition_1 = torch.tensor(Tonumpy(condition_l)).long()
        #condition = settings.DIF10.q_sample(condition,0)
        latent_vector0=torch.tensor(Tonumpy(latent0)).long()
        latent_vector0=latent_vector0.view(1,-1)
        latent_vector=latent_vector.view(1,-1)
        latent_vector2=latent_vector2.view(1,-1)
        latent_vector3=latent_vector3.view(1,-1)
        latent_vector4=latent_vector4.view(1,-1)
        latent_vector5=latent_vector5.view(1,-1)
        latent_vector6=latent_vector6.view(1,-1)
        latent_vector7=latent_vector7.view(1,-1)
        latent_vector8=latent_vector8.view(1,-1)
        latent_vector9=latent_vector9.view(1,-1)
       
        condition=condition.view(1,-1)
        #print(condition.shape)
        #output=settings.DIF1016.q_sample(condition,1)
        #output = settings.DIF1010cur.p_sample(condition,latent_vector)
        #output=settings.DIF1015ori.p_sample(output,latent_vector2,condition)
        #output = settings.DIF1016.p_sample(latent_vector9, latent_vector0, latent_vector9)  
        output = settings.DIF1016.p_sample(condition,latent_vector0,condition)
        #output = settings.DIF1016.p_sample(output, latent_vector0, latent_vector9)      
        output = settings.DIF1016.p_sample(output,latent_vector0,condition)
        output = settings.DIF1016.p_sample(output, latent_vector0, latent_vector9)
        output = settings.DIF1016.p_sample(output,latent_vector0,condition)
        
        output = output.detach().numpy()
        vae_output.ref()[:] = output
        #print(vae_output.shape)
        return vae_output
def test_dif1016(latent,latent2,latent3,latent4,latent5,latent6,latent7,latent8,latent9,condition_l,latent0,vae_output):
    with torch.no_grad():
        settings.DIF1016.eval()
        latent_vector = torch.tensor(Tonumpy(latent)).float()
        latent_vector2 = torch.tensor(Tonumpy(latent2)).float()
        latent_vector3 = torch.tensor(Tonumpy(latent3)).float()
        latent_vector4 = torch.tensor(Tonumpy(latent4)).float()
        latent_vector5 = torch.tensor(Tonumpy(latent5)).float()
        latent_vector6 = torch.tensor(Tonumpy(latent6)).float()
        latent_vector7 = torch.tensor(Tonumpy(latent7)).float()
        latent_vector8 = torch.tensor(Tonumpy(latent8)).float()
        latent_vector9 = torch.tensor(Tonumpy(latent9)).float()
        
        condition = torch.tensor(Tonumpy(condition_l)).float()
        #condition_1 = torch.tensor(Tonumpy(condition_l)).long()
        #condition = settings.DIF10.q_sample(condition,0)
        latent_vector0=torch.tensor(Tonumpy(latent0)).long()
        latent_vector0=latent_vector0.view(1,-1)
        latent_vector=latent_vector.view(1,-1)
        latent_vector2=latent_vector2.view(1,-1)
        latent_vector3=latent_vector3.view(1,-1)
        latent_vector4=latent_vector4.view(1,-1)
        latent_vector5=latent_vector5.view(1,-1)
        latent_vector6=latent_vector6.view(1,-1)
        latent_vector7=latent_vector7.view(1,-1)
        latent_vector8=latent_vector8.view(1,-1)
        latent_vector9=latent_vector9.view(1,-1)
       
        condition=condition.view(1,-1)
        #print(condition.shape)
        #output=settings.DIF1016.q_sample(condition,1)
        #output = settings.DIF1010cur.p_sample(condition,latent_vector)
        #output=settings.DIF1015ori.p_sample(output,latent_vector2,condition)
        output = settings.DIF1016.p_sample(latent_vector, latent_vector0, condition)
        output = settings.DIF1016.p_sample(output, latent_vector0, latent_vector2)
        output = settings.DIF1016.p_sample(output, latent_vector0, latent_vector3)
        output = settings.DIF1016.p_sample(output, latent_vector0, latent_vector4)
        output = settings.DIF1016.p_sample(output, latent_vector0,latent_vector5)
        output = settings.DIF1016.p_sample(output, latent_vector0, latent_vector6)
        output = settings.DIF1016.p_sample(output, latent_vector0, latent_vector7)
        output = settings.DIF1016.p_sample(output, latent_vector0, latent_vector8)
        output = settings.DIF1016.p_sample(output, latent_vector0, latent_vector9)
        output = settings.DIF1016.p_sample(output,latent_vector0,condition)
        
        output = output.detach().numpy()
        vae_output.ref()[:] = output
        #print(vae_output.shape)
        return vae_output

if __name__=="__main__":
    main()
