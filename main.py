import os
import cv2
import time
import numpy   as np
from   tqdm    import tqdm
from   pathlib import Path
from   PIL     import Image

import torch
from   torch            import nn
from   torch.utils.data import DataLoader, Dataset
from   torchvision      import transforms
 
import torchjpeg

# RDSR in Eq. (23)
class RDSR(nn.Module):

    def __init__(self, num_dnn):

        super(RDSR, self).__init__()

        # number of DNNs K in Eq. (23)
        self.num_dnn = num_dnn

        # layers in Table 2
        layers = []
        layers.append(nn.Conv2d( 1, 64, kernel_size=5, stride=1, padding=2, bias=True, padding_mode='replicate'))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=True, padding_mode='replicate'))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(32,  1, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate'))

        # $\phi_\theta$ in Eq. (23)
        all_layers = []
        all_layers.append(layers)
        self.dnn = nn.Sequential(*layers)

    def forward(self, dc, con_upp, con_low):

        out = dc
        
        for _ in range(self.num_dnn):

            # $\phi_\theta$ in Eq. (23)
            out = self.dnn(out)

            # POCS in Eq. (23)
            out = proj(out, con_upp, con_low)

        return out


# read images from 'img_dir' and convert images to dataset
class ImageFolder(Dataset):

    IMG_EXTENSIONS = [".png"]

    def __init__(self, img_dir, transform=None):

        self.img_paths = self._get_img_paths(img_dir)
        self.transform = transform

    def __getitem__(self, index):

        path = self.img_paths[index]

        img = Image.open(path)
        img = img.convert('L')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def _get_img_paths(self, img_dir):

        img_dir = Path(img_dir)
        img_paths = [
            p for p in img_dir.iterdir() if p.suffix in ImageFolder.IMG_EXTENSIONS
        ]
        img_paths.sort()

        return img_paths

    def __len__(self):
        return len(self.img_paths)
 

# projection operator in Eq. (19)
def proj(out_batch, con_upp, con_low):

    out_batch = out_batch * 255
    out_batch = torchjpeg.batch_dct(out_batch)
    out_batch = torch.minimum(out_batch, con_upp)
    out_batch = torch.maximum(out_batch, con_low)
    out_batch = torchjpeg.batch_idct(out_batch)
    out_batch = out_batch / 255

    return out_batch


# batch to numpy array
def batch_to_ndarray(batch):
 
    ndarray = batch.cpu().detach().numpy()
    ndarray = np.squeeze(ndarray)
    ndarray = ndarray * 255
    ndarray = np.round(ndarray)
    ndarray = np.maximum(ndarray, 0)
    ndarray = np.minimum(ndarray, 255)
    ndarray = np.array(ndarray, np.uint8)

    return ndarray
 

# compute BPS, BPP, AoS, and PSNR
def get_bps_bpp_aos_psnr(qtz_ind_jpg, jpg_batch, rec_batch):

    qtz_ind_jpg = qtz_ind_jpg.cpu().detach().numpy()
    qtz_ind_jpg = np.squeeze(qtz_ind_jpg)
    h, w = qtz_ind_jpg.shape

    ind_rec = torchjpeg.batch_dct(rec_batch * 255)
    ind_rec = ind_rec.cpu().detach().numpy()
    ind_rec = np.squeeze(ind_rec)

    # original sign of $y$ in Eq. (7)
    org_sgn = np.sign(qtz_ind_jpg)
    org_sgn[::8, ::8] = 0

    # retrieved sign of $c \tilde{x}$ in Eq. (7)
    rec_sgn = np.sign(ind_rec)
    rec_sgn[np.where(rec_sgn == 0)] = 1
    rec_sgn[::8, ::8] = 0
    rec_sgn[np.where(qtz_ind_jpg == 0)] = 0

    # residual $e$ in Eq. (7)
    bit_res = org_sgn * rec_sgn

    # probability of 0 and 1 in e
    num_pos = np.count_nonzero(bit_res == +1)
    num_neg = np.count_nonzero(bit_res == -1)
    num = num_pos + num_neg

    if num == 0:
        print('error')
        exit(1)

    prb_pos = num_pos / num
    prb_neg = num_neg / num

    # BPS, BPP, and AoS
    bps = -(prb_pos * np.log2(prb_pos) + prb_neg * np.log2(prb_neg))    
    bpp = bps * num / (h * w)
    aos = prb_pos * 100

    # PSNR
    jpg_ndarray = batch_to_ndarray(jpg_batch)
    rec_ndarray = batch_to_ndarray(rec_batch)
    psnr = cv2.PSNR(jpg_ndarray, rec_ndarray)
    
    return bps, bpp, aos, psnr, rec_ndarray


def training(device, net, criterion, optimizer, epochs, input_loader, org_set):
 
    for epoch in range(1, epochs+1):

        running_loss = 0.0

        for counter, input_batch in enumerate(tqdm(input_loader), 1):
            
            input_batch = input_batch.to(device)

            org_list = []
            
            for b in range(batch_size):
                org_list.append(org_set.__getitem__(batch_size*(counter-1)+b))
            
            org_batch = torch.stack(org_list)
            org_batch = org_batch.to(device)
            
            # DCT coefficient
            qtz_ind_jpg = torchjpeg.compress_coefficients(org_batch, train_QF, table="luma")
            qtz_ind_our = torch.abs(qtz_ind_jpg)
            deg_cff_our = torchjpeg.dequantize_at_quality(qtz_ind_our, train_QF, table="luma")
            
            # upper and lower bounds of constraint in Eqs. (14) and (16)
            con_upp =  deg_cff_our.clone()
            con_low = -deg_cff_our.clone()
            con_low[:, :, ::8, ::8] = -con_low[:, :, ::8, ::8]

            optimizer.zero_grad()
            output = net(input_batch, con_upp, con_low)
            
            loss = criterion(org_batch, output)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / counter
        print('epoch:{0:d}, loss:{1:f}'.format(epoch, avg_loss))

    torch.save(net.state_dict(), model_save_dir + '/' + model_name + '.pth')


# QF 
train_QF = 50
test_QF  = 50

# 0: test, 1: train
train_flag = 1

# learning rate
lr_opt = 0.0002

# epochs
EPOCHS = 50

# batch size
batch_size = 10

# number of networks K
num_dnn = 20

# model name and directory
model_name = 'RDSR'
model_save_dir = 'model'

# mkdir
os.makedirs(model_save_dir, exist_ok=True)

# define transform and device
transform = transforms.Compose([transforms.ToTensor()]) 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = RDSR(num_dnn) 
net = net.to(device)

# training
if train_flag:
 
    org_set      = ImageFolder('train/org', transform)
    input_set    = ImageFolder('train/dc/' + str(train_QF), transform)
    input_loader = DataLoader(input_set, batch_size=batch_size, shuffle=False)

    # set training flag in network
    net.train()

    # loss function in Eq. (22)
    criterion = torch.nn.MSELoss()

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_opt)

    training(device, net, criterion, optimizer, EPOCHS, input_loader, org_set) 
 
# test
else:

    with torch.no_grad():

        # path to model
        model_path = model_save_dir + '/' + model_name + '.pth'

        if not os.path.isfile(model_path):
            print('model not found')
            exit(1)

        # set test flag in network
        net.eval()

        # read network weights
        net.load_state_dict(torch.load(model_path, map_location=device))

        # dir for reconstructed images
        test_rec_dir = 'rec/test_QF=' + str(test_QF)
        os.makedirs(test_rec_dir, exist_ok=True)
        
        bpp_sum  = 0
        bps_sum  = 0
        aos_sum  = 0
        psnr_sum = 0
        time_sum = 0

        for img_num in tqdm(range(1, 61)):

            # read image
            org_gray = cv2.imread('test/%02d.png' % (img_num), cv2.IMREAD_GRAYSCALE)
            h, w = org_gray.shape

            # image to batch
            org_tensor = transform(org_gray)
            org_batch = org_tensor.unsqueeze(0).to(device)

            # quantization index
            qtz_ind_jpg = torchjpeg.compress_coefficients(org_batch, test_QF, table="luma")
            
            # magnitudes
            qtz_ind_our = torch.abs(qtz_ind_jpg)
            
            # dequantization
            deg_cff_our = torchjpeg.dequantize_at_quality(qtz_ind_our, test_QF, table="luma")

            # upper and lower bounds of constraint in Eqs. (14) and (16)
            con_upp = deg_cff_our.clone()
            con_low = -deg_cff_our.clone()
            con_low[:, :, ::8, ::8] = -con_low[:, :, ::8, ::8]

            # JPEG image
            jpg_batch = torchjpeg.decompress_coefficients(qtz_ind_jpg, test_QF, table="luma")

            # direct components
            qtz_ind_dc = torch.zeros((1, 1, h, w), dtype=torch.float32)
            qtz_ind_dc[:, :, ::8, ::8] = qtz_ind_jpg[:, :, ::8, ::8]
            input_batch = torchjpeg.decompress_coefficients(qtz_ind_dc, test_QF, table="luma")
            input_batch = input_batch.to(device)

            # execution time
            start_time = time.time()
            rec_batch = net(input_batch, con_upp, con_low)
            elapsed_time = time.time() - start_time

            # BPS, BPP, AoS, and PSNR
            bps, bpp, aos, psnr, rec_img = get_bps_bpp_aos_psnr(qtz_ind_jpg, jpg_batch, rec_batch)

            bps_sum  += bps
            bpp_sum  += bpp
            aos_sum  += aos
            psnr_sum += psnr
            time_sum += elapsed_time 

            cv2.imwrite(test_rec_dir + '/%02d.png' % img_num, rec_img)

        # results
        print('BPS  = {0:6.4f} [bit]'.format(bps_sum/60))
        print('BPP  = {0:6.4f} [bpp]'.format(bpp_sum/60))
        print('AoS  = {0:3.2f} [%]'.format(aos_sum/60))   
        print('PSNR = {0:3.2f} [dB]'.format(psnr_sum/60))
        print('Time = {0:6.4f} [s]'.format(time_sum/60))