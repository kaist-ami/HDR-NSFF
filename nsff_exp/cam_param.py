import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

img2mse = lambda x, y, z : torch.mean((x - y) ** 2 * z)

def set_initial_values(images):
    # images = images.numpy()
    linear_imgs = np.power(images, 2.2)
    overall_mean = np.mean(linear_imgs, axis=(0,1,2,3)) # mean
    per_image_mean = np.mean(linear_imgs, axis=(1,2,3)) # (N, )
    ref_idx = np.argmin(np.abs(np.subtract(per_image_mean,overall_mean))) # Minimum difference idx

    overall_channel_wise = np.mean(linear_imgs, axis=(0,1,2)) # (C, )
    per_image_channel_wise = np.mean(linear_imgs, axis=(1,2)) # (N, C, )

    ratio = per_image_channel_wise / overall_channel_wise # (N, C, )

    wb = np.log(ratio) # (N, C, )

    return torch.Tensor(wb), ref_idx

def set_fix_value(images):
    return np.mean(images, axis=(0,1,2,3))

def leaky_thresholding(Iv, is_train, leaky_values=0.01):
    _Iv = torch.empty_like(Iv).copy_(Iv)

    if is_train:
        add_over = leaky_values * Iv[Iv>1]
        denom = (Iv.abs() + 1e-4).sqrt()
        add_under = (-leaky_values / denom + leaky_values)[Iv<0]
        _Iv[Iv>1] = Iv[Iv>1]+add_over
        _Iv[Iv<0] = Iv[Iv<0]+add_under    
    
    else:
        _Iv[Iv>1] = 1
        _Iv[Iv<0] = 0
    
    return _Iv

def response_function(Iv, CRF, is_train, device, idx, leaky_values=0.01):
    Iv = Iv*2-1
    n, c = Iv.shape

    idx = idx * torch.ones(1)
    if idx.shape[0] != n :
        idx = idx.repeat(n)

    Ildr = torch.zeros(n,c) # N_rand, 3
    
    leak_add = torch.zeros_like(Ildr).to(device) # N_rand,3
    
    if is_train:
        leak_add = leak_add + torch.where(Iv>1, leaky_values*Iv, leak_add)
        leak_add = leak_add + torch.where(Iv<-1, (-leaky_values / ((Ildr.abs() + 1e-4).sqrt()) + leaky_values).to(device), leak_add)
    
    for i in range(c):
        # response_sl = CRF[:,:,i].view(1,1,CRF.shape[0],-1)
        # zeros = torch.zeros(Iv.shape[0]).to(device)
        # _idx = idx.to(device)
        # sl = torch.cat((Iv[:,i].unsqueeze(-1), _idx.unsqueeze(-1)),axis=1)
        # sl = sl.view(1,1,n,2)
        # Ildr[:,i] = torch.nn.functional.grid_sample(response_sl, sl, padding_mode='border', align_corners=True)[0,0,0,:]
        response_sl = CRF[:,i].view(1,1,1,-1)
        zeros = torch.zeros(Iv.shape[0]).to(device)
        sl = torch.cat((Iv[:,i].unsqueeze(-1), zeros.unsqueeze(-1)),axis=1)
        sl = sl.view(1,1,n,2)
        Ildr[:,i] = torch.nn.functional.grid_sample(response_sl, sl, padding_mode='border', align_corners=True)[0,0,0,:]
        
    if is_train==True:
        LDR = Ildr.to(device) + leak_add
    else :
        LDR = Ildr.to(device)

    LDR = (LDR+1)/2
    
    return LDR

# base - in the paper
class rgb_network(nn.Module):
    def __init__(self, input_rad=3, split=True):
        super(rgb_network, self).__init__()
        self.split = split
        if split:
            self.feature_linear_1_r = nn.Linear(1, 512)
            self.feature_linear_2_r = nn.Linear(512, 1)
            
            self.feature_linear_1_g = nn.Linear(1, 512)
            self.feature_linear_2_g = nn.Linear(512, 1)

            self.feature_linear_1_b = nn.Linear(1, 512)
            self.feature_linear_2_b = nn.Linear(512, 1)
            self.sigmoid = nn.Sigmoid()
            
        else:
            self.feature_linear_1 = nn.Linear(input_rad, 512)
            self.feature_linear_2 = nn.Linear(512, 3)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.split:
            rgb_r = self.feature_linear_1_r(x[:,0].unsqueeze(1))
            rgb_r = self.feature_linear_2_r(rgb_r)
            rgb_r = self.sigmoid(rgb_r)

            rgb_g = self.feature_linear_1_g(x[:,1].unsqueeze(1))
            rgb_g = self.feature_linear_2_g(rgb_g)
            rgb_g = self.sigmoid(rgb_g)

            rgb_b = self.feature_linear_1_b(x[:,2].unsqueeze(1))
            rgb_b = self.feature_linear_2_b(rgb_b)
            rgb_b = self.sigmoid(rgb_b)

            rgb = torch.cat((rgb_r, rgb_g, rgb_b), 1)

            return rgb
        else :
            rgb = self.feature_linear_1(x)
            rgb = self.feature_linear_2(rgb)
            rgb = self.sigmoid(rgb)

            return rgb

# base - in the paper
class only_sigmoid(nn.Module):
    def __init__(self, input_rad=3):
        super(only_sigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.sharpness = 10

    def forward(self, x):
        
        x = (x-0.5) * self.sharpness
        rgb = self.sigmoid(x)
        return rgb

class only_gamma(nn.Module):
    def __init__(self, gamma=2.2):
        super(only_gamma, self).__init__()
        self.gamma = gamma

    def forward(self, x):
        x = torch.clamp(x, min=1e-6, max=1.0)
        rgb = x.pow(1.0 / self.gamma)
        return rgb

class CamParam:
    def __init__(self, N, H, W, device, gts, initialize=True, tone_mapping="piece_wise", share_crf=False, share_wb=False, ref_idx=0, log_scale=False):
        self.device = device
        self.tone_mapping = tone_mapping
        self.share_crf = share_crf
        self.share_wb = share_wb
        self.N = N
        self.log_scale = log_scale
        # learnable white-balance Tensor : N*3
        # initialize with 0 => used as exp(white_balance) > 0
        white_balance_init = torch.zeros((N,3), dtype=torch.float32)    # [N,3]
        self.wb = nn.Embedding.from_pretrained(white_balance_init, freeze=False).to(self.device)

        # learnable CRF Tensor : N*256*3
        # initialize as x^(2.2) (x in [0,1])
        if self.tone_mapping == "piece_wise":
            CRF_init = torch.arange(1/256, 1, 1/256).repeat(3).view(3,255).permute(1,0)
            CRF_init = torch.pow(CRF_init, 1/2.2)
            CRF_init = CRF_init*2 - 1
            CRF_init = CRF_init.unsqueeze(0).repeat(N,1,1) # [N,255,3]
            CRF_init = CRF_init.requires_grad_()
            CRF_init = CRF_init.to(self.device)

            class crf_module(torch.nn.Module):
                def __init__(self, CRF_init):
                    super(crf_module, self).__init__()
                    self.alpha = nn.Parameter(CRF_init)

            self.CRF = crf_module(CRF_init)
            
        elif self.tone_mapping == "share":
            CRF_init = torch.arange(1/256, 1, 1/256).repeat(3).view(3,255).permute(1,0)
            CRF_init = torch.pow(CRF_init, 1/2.2)
            CRF_init = CRF_init*2 - 1
            CRF_init = CRF_init.unsqueeze(0).repeat(1,1,1) # [1,255,3]
            CRF_init = CRF_init.requires_grad_()
            CRF_init = CRF_init.to(self.device)
            
            class crf_module(torch.nn.Module):
                def __init__(self, CRF_init):
                    super(crf_module, self).__init__()
                    self.alpha = nn.Parameter(CRF_init)
                    
            self.CRF = crf_module(CRF_init)
        
        elif self.tone_mapping == "nn":
            split_channel = True
            self.rgb_model = rgb_network(split=split_channel).to(device)
            self.CRF = None
        elif self.tone_mapping == "hdr_hexplane":
            # self.rgb_model = only_sigmoid().to(device)
            self.rgb_model = only_gamma().to(device)
            self.CRF=None
        # Set initial exp, wb values
        if initialize :
            #TODO: choose reference image from training close to mean value
            white_balance_init, self.ref_idx = set_initial_values(gts)
            self.wb = nn.Embedding.from_pretrained(white_balance_init, freeze=False).to(self.device)            
        else : 
            #TODO: if not initialize, then ref_idx = 0
            self.ref_idx = ref_idx
        
        ref_wb = self.wb.weight[self.ref_idx].detach().cpu().numpy()
        self.ref_wb = torch.exp(torch.Tensor(ref_wb)).to(self.device)   # Initialize=False 이기 때문, idx 1의 white balance를 ref_wb로 지정.
        
        # find boundary from gt dimension
        N, W, H, C = gts.shape
        self.size = (W,H)
        self.num = N
        bounds = []
        for i in range(N+1):
            bounds.append(i*W*H)
        self.boundary = torch.tensor(bounds)

        self._frozen_forever = False    # eternal freeze
        self._frozen_iter = False       # just freeze one step for difix

    @torch.no_grad()
    def set_trainable(self, on: bool):
        if hasattr(self, "wb") and self.wb is not None:
            for p in self.wb.parameters():
                p.requires_grad = on
        if hasattr(self, "CRF") and self.CRF is not None:
            for p in self.CRF.parameters():
                p.requires_grad = on

        # self.train(on)

    @torch.no_grad()
    def freeze_for_iter(self):
        self._frozen_iter = True
        self.set_trainable(False)
    
    @torch.no_grad()
    def unfreeze_for_iter(self):
        self._frozen_iter = False
        if not self._frozen_forever:
            self.set_trainable(True)

    @torch.no_grad()
    def freeze_forever(self):
        self._frozen_forever = True
        self.set_trainable(False)

    def is_frozen_forever(self) -> bool:
        return bool(self._frozen_forever)
    
    def is_frozen_this_iter(self) -> bool:
        return bool(self._frozen_iter)
        
    def optimizer(self, l_rate):
        # grad_vars = list(self.wb.parameters())
        wb_params = list(self.wb.parameters())
        # crf_params = list(self.CRF.parameters()) if self.tone_mapping == "piece_wise" else []

        if self.tone_mapping == "piece_wise":
            self.crf_params = list(self.CRF.parameters())
            if self.share_crf:
                # optimizer_crf = None
                optimizer_crf = torch.optim.Adam(params=self.crf_params, lr=l_rate / self.N , betas=(0.9, 0.999))  # CRF에 다른 lr 적용
                # optimizer_wb = torch.optim.Adam(params=wb_params, lr=(l_rate / self.N * 3), betas=(0.9, 0.999))
                optimizer_wb = torch.optim.Adam(params=wb_params, lr=l_rate, betas=(0.9, 0.999))
            else:
                # optimizer_crf = None
                optimizer_crf = torch.optim.Adam(params=self.crf_params, lr=l_rate, betas=(0.9, 0.999))
                optimizer_wb = torch.optim.Adam(params=wb_params, lr=l_rate, betas=(0.9, 0.999))
        elif self.tone_mapping == "nn":
            nn_params = list(self.rgb_model.parameters())
            optimizer_crf = torch.optim.Adam(params=nn_params, lr=l_rate, betas=(0.9, 0.999))
            optimizer_wb = torch.optim.Adam(params=wb_params, lr=l_rate, betas=(0.9, 0.999))
        elif self.tone_mapping == "hdr_hexplane":
            nn_params = list(self.rgb_model.parameters())
            optimizer_crf = None
            optimizer_wb = torch.optim.Adam(params=wb_params, lr=l_rate, betas=(0.9, 0.999))
        # optimizer = torch.optim.Adam(params=grad_vars, lr=l_rate, betas=(0.9, 0.999))
        # optimizer = torch.optim.RMSprop(params=grad_vars, lr = l_rate, alpha=0.95)
        # return optimizer
        return optimizer_wb, optimizer_crf
        
    def find_index(self, indexer):
        self.cam_index = torch.add(torch.bucketize(indexer, self.boundary, right=True),-1)

        V = indexer // (self.size[0]*self.size[1])
        x = (indexer-V*(self.size[0]*self.size[1])) % self.size[0]
        y = (indexer-V*(self.size[0]*self.size[1])) // self.size[0]

        self.cam_index_coords = torch.cat((x.unsqueeze(1), y.unsqueeze(1)), 1).to(self.device)

    def rad2ldr(self, HDR, white_balance, CRF, is_train, device, idx):
        if is_train == True:
            # import pdb; pdb.set_trace()
            # ref_idx = (idx == self.ref_idx).nonzero(as_tuple=True)[0]
            # non_ref_idx = (idx != self.ref_idx).nonzero(as_tuple=True)[0]
            # ref_idx = (idx == self.ref_idx).nonzero()[0]
            # non_ref_idx = (idx != self.ref_idx).nonzero()[0]
            if self.log_scale:
                Iw = torch.zeros_like(HDR)
                if (idx == self.ref_idx)==False:
                    Iw = HDR + torch.log(white_balance)
                    relu = nn.ReLU()
                    Iw = relu(Iw)
                else:
                    Iw = HDR + torch.log(self.ref_wb)
                    relu = nn.ReLU()
                    Iw = relu(Iw)
            else:
                Iw = torch.zeros_like(HDR)
                if (idx == self.ref_idx)==False:
                    Iw = HDR * white_balance
                else:
                    Iw = HDR * self.ref_wb            
        else:
            if self.log_scale:
                Iw = HDR + torch.log(white_balance)
            else:
                Iw = HDR * white_balance
        
        if self.tone_mapping == "piece_wise":
            LDR = response_function(Iw, CRF, is_train, device, idx)
        elif self.tone_mapping == "nn":
            LDR = self.rgb_model(Iw)
        elif self.tone_mapping == "hdr_hexplane":
            LDR = self.rgb_model(Iw)
        return LDR

    def RAD2LDR(self, rad_pred, idx, is_train, detach_param=False):
        
        crf_idx = idx
        wb_idx = idx

        if self.tone_mapping == "piece_wise":
            
            if self.share_crf:
                crf_idx = 0

                if self.share_wb:
                    wb_idx = idx % 3
                
            N_CRF = list(self.CRF.parameters())[0][crf_idx,:,:].to(self.device)
            N_CRF = torch.cat((-1*torch.ones(1,3, requires_grad=False).to(self.device), N_CRF, torch.ones(1, 3, requires_grad=False).to(self.device)), dim=0)
            N_wb = torch.exp(self.wb.weight[wb_idx])

        # elif self.tone_mapping == "share":
        #     N_CRF = list(self.CRF.parameters())[0][0,:,:].to(self.device)
        #     N_CRF = torch.cat((-1*torch.ones(1,3, requires_grad=False).to(self.device), N_CRF, torch.ones(1, 3, requires_grad=False).to(self.device)), dim=0)
        elif self.tone_mapping == "hdr_hexplane":
            N_CRF=None
            N_wb = torch.exp(self.wb.weight[wb_idx])
        
        elif self.tone_mapping == "nn":
            N_CRF = None
            N_wb = torch.exp(self.wb.weight[wb_idx])
            
        else :
            N_CRF = None
            N_wb = None

        if detach_param:
            if N_CRF is not None: N_CRF = N_CRF.detach()
            if N_wb is not None: N_wb = N_wb.detach()

        ldr_pred = self.rad2ldr(rad_pred, N_wb, N_CRF, is_train, self.device, wb_idx)

        return ldr_pred
    
    def RAD2LDR_img(self, rad_img, idx):
        is_train = False
        
        H, W, _ = rad_img.shape        
        rad_img = rad_img.reshape(H*W, -1)
        
        crf_idx = idx
        wb_idx = idx

        if self.tone_mapping == "piece_wise":
            if self.share_crf:
                crf_idx = 0
                if self.share_wb:
                    wb_idx = idx % 3
                
            N_CRF = list(self.CRF.parameters())[0][crf_idx,:,:].to(self.device) # list(self.CRF.parameters())[0].shape : [N_imgs, 255, 3]
            N_CRF = torch.cat((-1*torch.ones(1,3, requires_grad=False).to(self.device), N_CRF, torch.ones(1, 3, requires_grad=False).to(self.device)), dim=0)
            # N_CRF = N_CRF.unsqueeze(0)
            N_wb = torch.exp(self.wb.weight[wb_idx])
            
        elif self.tone_mapping == "hdr_hexplane":
            N_CRF=None
            N_wb = torch.exp(self.wb.weight[wb_idx])
        
        elif self.tone_mapping == "nn":
            N_CRF = None
            N_wb = torch.exp(self.wb.weight[wb_idx])
        
        else :
            N_CRF = None
            N_wb = None

        ldr_img = self.rad2ldr(rad_img, N_wb, N_CRF, is_train, self.device, wb_idx)
        # ldr_img = self.rad2ldr(rad_img, N_wb, N_CRF, is_train, self.device, idx.unsqueeze(0))
        ldr_img = ldr_img.reshape(H,W,3)

        return ldr_img

    def RAD2LDR_img_control(self, rad_img, idx):
        is_train = False
        H, W, _ = rad_img.shape
        
        rad_img = rad_img.reshape(H*W, -1)
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1)
        coords = coords.reshape(H*W, -1).to(self.device)
                
        N_wb = torch.exp(self.wb.weight[idx:idx+1])

        N_CRF = list(self.CRF.parameters())[0][idx,:,:].to(self.device)
        N_CRF = torch.cat((-1*torch.ones(1,3, requires_grad=False).to(self.device), N_CRF, torch.ones(1, 3, requires_grad=False).to(self.device)), dim=0)
        N_CRF = N_CRF.unsqueeze(0)

        ldr_img = self.rad2ldr(rad_img, N_wb, N_CRF, is_train, self.device, idx.unsqueeze(0))
        ldr_img = ldr_img.reshape(H,W,3)

        return ldr_img
    
    def crf_smoothness_loss(self):
        if self.tone_mapping=="piece_wise":
            CRF = list(self.CRF.parameters())[0][:,:,:].to(self.device)
            CRF = torch.cat((-1*torch.ones((len(CRF),1,3), requires_grad=False).to(self.device), \
                            CRF, torch.ones((len(CRF),1, 3), requires_grad=False).to(self.device)), dim=1)

            g_pp = -2*CRF[:, 1:256, :] + CRF[:,:255,:] + CRF[:,2:257,:]

            crf_loss = torch.sum(torch.square(g_pp))
        else :
            crf_loss = torch.tensor([0]).to(self.device)

        return crf_loss
    
    def save(self, path):
        if self.tone_mapping == "piece_wise":
            torch.save({
                # (kjs) add parameters to be saved
                'white_balance': self.wb.state_dict(),
                'CRF': self.CRF.state_dict(),
            }, path)
        elif self.tone_mapping == "nn":
            torch.save({
                'white_balance': self.wb.state_dict(),
                'CRF': None,
                'rgb_model' : self.rgb_model.state_dict(),
            })
            
        elif self.tone_mapping == "hdr_hexplane":
            torch.save({
                'white_balance': self.wb.state_dict(),
                'CRF': None,
            })
    
    def save_txt(self, dir):
        
        white_balance = torch.exp(self.wb.weight).detach().cpu().numpy()
        
        np.savetxt(dir+"/wb.txt", white_balance)

        if self.tone_mapping == "piece_wise":
            CRF = list(self.CRF.parameters())[0]
            CRF = torch.cat((-1*torch.ones(CRF.shape[0], 1, 3, requires_grad=False).to(self.device), CRF, torch.ones(CRF.shape[0], 1, 3, requires_grad=False).to(self.device)), dim=1)
            CRF = (torch.add(CRF, 1)/2)
            cam_resp_func = CRF.detach().cpu().numpy()
            
            i = 0
            with open(dir+"/crf.txt", 'w') as outfile:
                    # I'm writing a header here just for the sake of readability
                    # Any line starting with "#" will be ignored by numpy.loadtxt
                    outfile.write('number of camera: {0}\n'.format(i))

                    # Iterating through a ndimensional array produces slices along
                    # the last axis. This is equivalent to data[i,:,:] in this case
                    for data_slice in cam_resp_func:
                        i = i+1
                        # The formatting string indicates that I'm writing out
                        # the values in left-justified columns 7 characters in width
                        # with 2 decimal places.  
                        np.savetxt(outfile, data_slice.T, fmt='%-7.2f')

                        # Writing out a break to indicate different slices...
                        outfile.write('number of camera: {0}\n'.format(i))
        
            # np.savetxt(dir+"/crf.txt", cam_resp_func)
            
            x = np.arange(0,1+1/256, 1/256)
            plt.plot(np.log(x), cam_resp_func[0]) #FIXME: need to fix this 어쩐지 이상하더라!
            plt.xlabel('log Exposure')
            plt.savefig(dir+"/sample_crf.png")

    def load_ckpt(self, ckpt):
        print(ckpt)
        # print(ckpt['white_balance'])
        # self.wb.load_state_dict(torch.load(ckpt)['white_balance'])
        self.wb.load_state_dict(ckpt['white_balance'])
        if self.tone_mapping == "piece_wise":
            # self.CRF.load_state_dict(torch.load(ckpt)['CRF'])
            self.CRF.load_state_dict(ckpt['CRF'])
        elif self.tone_mapping == 'nn':
            self.rgb_model.load_state_dict(ckpt['rgb_model'])
    
    
    def visualize_crf(self, idx=0, save_path="CRF_plot.png"):
        """
        cam_param: CamParam 클래스 객체
        idx      : 어떤 index의 CRF를 보여줄지 (N>1인 경우)
        """
        # cam_param.CRF.alpha => shape: (N, 255, 3)
        #  -1~1 값이므로, 0~1 범위로 되돌려야 시각적으로 이해하기 편함
        crf_lut = (self.CRF.alpha[idx] + 1) / 2.0  # shape: (255, 3)
        
        # x축은 1/256부터 1까지 1/256 간격 => 실제론 1/256 ~ 255/256 => 255개
        # (원 코드에서 arange(1/256, 1, 1/256)을 사용했으므로 여기에 맞춤)
        x = np.linspace(1/256, 255/256, 255)
        
        crf_lut_cpu = crf_lut.detach().cpu().numpy()  # (255, 3)
        
        plt.figure(figsize=(6, 4))
        plt.plot(x, crf_lut_cpu[:, 0], 'r-', label='R channel')
        plt.plot(x, crf_lut_cpu[:, 1], 'g-', label='G channel')
        plt.plot(x, crf_lut_cpu[:, 2], 'b-', label='B channel')
        
        plt.title(f"Learned CRF (index={idx})")
        plt.xlabel("Input (normalized 0~1)")
        plt.ylabel("Output (after CRF, 0~1)")
        plt.ylim([0, 1])
        plt.grid(True)
        plt.legend()
        
        # 파일로 저장
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 메모리 해제

    def get_visualize_crf(self, idx=0):
        """
        cam_param: CamParam 클래스 객체
        idx      : 어떤 index의 CRF를 보여줄지 (N>1인 경우)
        """
        # cam_param.CRF.alpha => shape: (N, 255, 3)
        #  -1~1 값이므로, 0~1 범위로 되돌려야 시각적으로 이해하기 편함
        crf_lut = (self.CRF.alpha[idx] + 1) / 2.0  # shape: (255, 3)
        
        # x축은 1/256부터 1까지 1/256 간격 => 실제론 1/256 ~ 255/256 => 255개
        # (원 코드에서 arange(1/256, 1, 1/256)을 사용했으므로 여기에 맞춤)
        x = np.linspace(1/256, 255/256, 255)
        
        crf_lut_cpu = crf_lut.detach().cpu().numpy()  # (255, 3)
        
        fig = plt.figure(figsize=(6, 4))
        plt.plot(x, crf_lut_cpu[:, 0], 'r-', label='R channel')
        plt.plot(x, crf_lut_cpu[:, 1], 'g-', label='G channel')
        plt.plot(x, crf_lut_cpu[:, 2], 'b-', label='B channel')
        
        plt.title(f"Learned CRF (index={idx})")
        plt.xlabel("Input (normalized 0~1)")
        plt.ylabel("Output (after CRF, 0~1)")
        plt.ylim([0, 1])
        plt.grid(True)
        plt.legend()
        return fig
    
    def get_visualize_sigmoid(self):
        """
        tone_mapping == 'hdr_hexplane'에서 사용되는 sigmoid function 시각화용
        실제 학습 중 사용되는 self.rgb_model.sigmoid를 기반으로 시각화합니다.
        """
        assert self.tone_mapping == "hdr_hexplane", "This visualization is only valid for hdr_hexplane mode."
        
        # 입력 범위: HDR 값은 [0,1] -> [-1,1] 정규화됨
        x = np.linspace(0, 1, 256)
        # x_scaled = x * 2 - 1  # [-1, 1]

        # tensor 변환 및 모델 통과
        x_tensor = torch.tensor(x).float().unsqueeze(1).to(self.device)  # shape: (256, 1)
        y_tensor = self.rgb_model(x_tensor)  # only_sigmoid 사용
        y = y_tensor.detach().cpu().numpy().squeeze()  # shape: (256,)

        # 시각화
        fig = plt.figure(figsize=(6, 4))
        plt.plot(x, y, 'k-', label='Sigmoid (from model)')
        plt.title("Sigmoid Tone Mapping (Used in hdr_hexplane)")
        plt.xlabel("Input (normalized 0~1)")
        plt.ylabel("Output (after sigmoid)")
        plt.ylim([0, 1])
        plt.grid(True)
        plt.legend()
    
        return fig
    
    def get_visualize_gamma(self):
        """
        tone_mapping == 'hdr_hexplane'에서 사용되는 sigmoid function 시각화용
        실제 학습 중 사용되는 self.rgb_model.sigmoid를 기반으로 시각화합니다.
        """
        assert self.tone_mapping == "hdr_hexplane", "This visualization is only valid for hdr_hexplane mode."
        
        # 입력 범위: HDR 값은 [0,1] -> [-1,1] 정규화됨
        x = np.linspace(0, 1, 256)
        # x_scaled = x * 2 - 1  # [-1, 1]

        # tensor 변환 및 모델 통과
        x_tensor = torch.tensor(x).float().unsqueeze(1).to(self.device)  # shape: (256, 1)
        y_tensor = self.rgb_model(x_tensor)  # only_sigmoid 사용
        y = y_tensor.detach().cpu().numpy().squeeze()  # shape: (256,)

        # 시각화
        fig = plt.figure(figsize=(6, 4))
        plt.plot(x, y, 'k-', label='Gamma')
        plt.title("Fixed Tone Mapping (Used in hdr_hexplane)")
        plt.xlabel("Input (normalized 0~1)")
        plt.ylabel("Output")
        plt.ylim([0, 1])
        plt.grid(True)
        plt.legend()
    
        return fig
    
    def get_visualize_crf_nn(self):
        """
        시그모이드 기반 MLP tone-mapping 모델에서 학습된 CRF-like function을 시각화합니다.
        - input: rgb_model (rgb_network instance)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import torch

        rgb_model = self.rgb_model.to(self.device).eval()

        # 입력 x: [0, 1] → [-1, 1] 정규화 (HDR radiance → tone-mapped LDR처럼)
        x = torch.linspace(0, 1, 256).to(self.device)
        # x_scaled = x * 2 - 1  # shape: [256]
        
        # per-channel 처리
        with torch.no_grad():
            if rgb_model.split:
                x_r = x.unsqueeze(1)
                x_g = x.unsqueeze(1)
                x_b = x.unsqueeze(1)

                y_r = rgb_model.feature_linear_2_r(rgb_model.feature_linear_1_r(x_r))
                y_g = rgb_model.feature_linear_2_g(rgb_model.feature_linear_1_g(x_g))
                y_b = rgb_model.feature_linear_2_b(rgb_model.feature_linear_1_b(x_b))

                y_r = rgb_model.sigmoid(y_r).squeeze().cpu().numpy()
                y_g = rgb_model.sigmoid(y_g).squeeze().cpu().numpy()
                y_b = rgb_model.sigmoid(y_b).squeeze().cpu().numpy()
            else:
                x_in = torch.stack([x, x, x], dim=1)  # [256, 3]
                y_out = rgb_model(x_in).cpu().numpy()
                y_r, y_g, y_b = y_out[:, 0], y_out[:, 1], y_out[:, 2]

        x_np = x.cpu().numpy()

        # 시각화
        fig = plt.figure(figsize=(6, 4))
        plt.plot(x_np, y_r, 'r-', label='R channel')
        plt.plot(x_np, y_g, 'g-', label='G channel')
        plt.plot(x_np, y_b, 'b-', label='B channel')
        plt.title("Learned CRF (MLP-based tone mapping)")
        plt.xlabel("Input (normalized radiance, [0~1])")
        plt.ylabel("Output (LDR after sigmoid)")
        plt.grid(True)
        plt.ylim([0, 1])
        plt.legend()
        return fig

    def save_wb_plot(self, save_path="wb_plot.png"):
        """
        cam_param: CamParam 클래스 객체
            - cam_param.wb.weight => shape: (N, 3) 
            - 내부 파라미터는 log 형태이므로, 실제 WB 값은 exp(wb.weight).
        save_path: 결과 이미지를 저장할 경로 (예: "wb_plot.png")
        """

        # 1) wb.weight: [N, 3] 형태 텐서(log 공간). 로그값이므로 exp() 취해야 실제 WB 곱셈 팩터가 됨
        wb_log = self.wb.weight.detach().cpu().numpy()  # shape: (N, 3)
        wb_val = np.exp(wb_log)  # 실제 WB 곱셈 팩터 (R/G/B)

        # 2) x축: 인덱스 (0 ~ N-1)
        x = np.arange(wb_val.shape[0])  # [0, 1, 2, ..., N-1]

        # 3) 플롯 그리기
        plt.figure(figsize=(6,4))
        plt.scatter(x, wb_val[:, 0], color='red', label='WB: R', alpha=0.7)
        plt.scatter(x, wb_val[:, 1], color='green', label='WB: G', alpha=0.7)
        plt.scatter(x, wb_val[:, 2], color='blue', label='WB: B', alpha=0.7)
        # plt.scatter(x, wb_val[:, 0], 'r-', label='WB: R')
        # plt.scatter(x, wb_val[:, 1], 'g-', label='WB: G')
        # plt.scatter(x, wb_val[:, 2], 'b-', label='WB: B')

        plt.title("White Balance (exponential scale)")
        plt.xlabel("Index (e.g. image or camera ID)")
        plt.ylabel("White Balance Value (Multiplicative Factor)")
        plt.legend()
        plt.grid(True)

        # 4) 파일로 저장 (dpi=300: 해상도, bbox_inches='tight': 여백 제거)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 사용 완료 후 figure 메모리 해제

    def get_wb_plot(self):
        """
        cam_param: CamParam 클래스 객체
            - cam_param.wb.weight => shape: (N, 3) 
            - 내부 파라미터는 log 형태이므로, 실제 WB 값은 exp(wb.weight).
        save_path: 결과 이미지를 저장할 경로 (예: "wb_plot.png")
        """

        # 1) wb.weight: [N, 3] 형태 텐서(log 공간). 로그값이므로 exp() 취해야 실제 WB 곱셈 팩터가 됨
        wb_log = self.wb.weight.detach().cpu().numpy()  # shape: (N, 3)
        wb_val = np.exp(wb_log)  # 실제 WB 곱셈 팩터 (R/G/B)

        # 2) x축: 인덱스 (0 ~ N-1)
        x = np.arange(wb_val.shape[0])  # [0, 1, 2, ..., N-1]

        # 3) 플롯 그리기
        fig = plt.figure(figsize=(6,4))
        plt.scatter(x, wb_val[:, 0], color='red', label='WB: R', alpha=0.7)
        plt.scatter(x, wb_val[:, 1], color='green', label='WB: G', alpha=0.7)
        plt.scatter(x, wb_val[:, 2], color='blue', label='WB: B', alpha=0.7)
        # plt.scatter(x, wb_val[:, 0], 'r-', label='WB: R')
        # plt.scatter(x, wb_val[:, 1], 'g-', label='WB: G')
        # plt.scatter(x, wb_val[:, 2], 'b-', label='WB: B')

        plt.title("White Balance (exponential scale)")
        plt.xlabel("Index (e.g. image or camera ID)")
        plt.ylabel("White Balance Value (Multiplicative Factor)")
        plt.legend()
        plt.grid(True)

        return fig
    
    def get_crf_stretch_value(self, eps=1e-4):
        
        assert self.tone_mapping == "piece_wise", "Only applicable for piece-wise tone mapping"
        assert self.share_crf, "Only applicable when CRF is shared"

        # 1. CRF 평균 R/G/B 각각 가져오기
        crf = (self.CRF.alpha[0] + 1) / 2.0  # shape: [255, 3], in [0,1]
        crf_np = crf.detach().cpu().numpy()     # numpy로 변환

        # 2. 각 채널이 1-eps 이상이 되는 가장 작은 index 찾기
        threshold = 1 - eps
        x = np.linspace(1/256, 255/256, 255)

        reach_idx = []
        for ch in range(3):
            above = np.where(crf_np[:, ch] >= threshold)[0]
            if len(above) > 0:
                idx = above[0]
                reach_idx.append(x[idx])
            else:
                reach_idx.append(1.0)  # 그냥 안 도달했으면 끝값

        # 3. 가장 늦게 도달하는 채널 기준으로 결정
        stretch_value = max(reach_idx)
        return stretch_value
    
    def crf_lowrank_loss(self, weight=1.0):
        if self.tone_mapping != "piece_wise":
            return torch.tensor(0.0, device=self.device)

        crfs = list(self.CRF.parameters())[0]  # [N, 255, 3]
        loss = 0.0
        for i in range(crfs.shape[0]):
            A = crfs[i] - crfs[i].mean(dim=0, keepdim=True)

            # ⚠️ 안정성을 위해 SVD만 CPU에서 수행
            A_cpu = A.detach().cpu().double()
            U, S, Vh = torch.linalg.svd(A_cpu, full_matrices=False)
            S = S.to(self.device)  # 다시 device로 복귀

            loss += torch.sum(S[1:] ** 2)
        loss = loss / crfs.shape[0]
        return loss * weight
    
    def crf_interframe_loss(self, weight=1.0):
        """
        Frame 간 CRF consistency 제약 (inner-product 기반)
        모든 frame의 CRF가 유사한 방향을 갖도록 -2 f_i^T f_j 를 minimize
        즉, cross-correlation을 최대화하는 형태
        """
        if self.tone_mapping != "piece_wise":
            return torch.tensor(0.0, device=self.device)

        crfs = list(self.CRF.parameters())[0]  # [N, 255, 3]
        N = crfs.shape[0]

        # flatten to [N, D] where D = 255*3
        f = crfs.reshape(N, -1)
        # normalize to avoid scale effect (cosine-like)
        f = f - f.mean(dim=1, keepdim=True)
        f = f / (f.norm(dim=1, keepdim=True) + 1e-8)

        # pairwise inner products f_i^T f_j
        G = torch.matmul(f, f.T)  # [N, N] similarity matrix

        # -2 f_i^T f_j 를 minimize == f_i^T f_j 를 maximize
        # self-similarity(i==j)는 제외
        mask = ~torch.eye(N, dtype=bool, device=self.device)
        pairwise_sum = torch.sum(G[mask]) / (N*(N-1))  # 평균 cross similarity

        # 내적을 최대화 → -pairwise_sum을 minimize
        loss = -pairwise_sum

        return loss * weight

# For Debugging
import copy
def cam_fp(cam):
    """Cam_param의 파라미터 fingerprint(dict)를 반환: {name: (sum, norm, shape)}"""
    out = {}
    # WB
    for n,p in cam.wb.named_parameters():
        t = p.detach()
        out[f'wb.{n}'] = (t.sum().item(), t.norm().item(), tuple(t.shape))
    # CRF (있으면)
    if hasattr(cam, 'CRF') and (cam.CRF is not None):
        for n,p in cam.CRF.named_parameters():
            t = p.detach()
            out[f'crf.{n}'] = (t.sum().item(), t.norm().item(), tuple(t.shape))
    return out

def cam_diff(fp_prev, fp_curr, atol=0.0):
    """두 fingerprint 비교해 바뀐 key만 돌려줌"""
    changed = {}
    for k in fp_prev:
        a, b = fp_prev[k], fp_curr[k]
        # (sum, norm) 중 하나라도 달라졌으면 변경으로 간주(허용오차 있으면 atol로)
        if (abs(a[0]-b[0])>atol) or (abs(a[1]-b[1])>atol):
            changed[k] = (a, b)
    return changed

def req_grad(cam):
    """requires_grad/grad 유무 한눈에 보기"""
    wb = [(n, p.requires_grad, p.grad is None) for n,p in cam.wb.named_parameters()]
    if hasattr(cam,'CRF') and (cam.CRF is not None):
        crf = [(n, p.requires_grad, p.grad is None) for n,p in cam.CRF.named_parameters()]
    else:
        crf = []
    return wb, crf