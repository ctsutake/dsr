import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
 
import os
import time
import copy as cp
from tqdm import tqdm
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
 
import torchjpeg



# FDSRモデル
class FDSR(nn.Module):

    def __init__(self, sr_block_num):
        super(FDSR, self).__init__()

        self.sr_block_num = sr_block_num

        layers = []
        layers.append(nn.Conv2d( 1, 64, kernel_size=5, stride=1, padding=2, bias=True, padding_mode='replicate'))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=True, padding_mode='replicate'))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(32,  1, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate'))

        all_layers = []
        all_layers.append(layers)
        for i in range(sr_block_num-1):
            l = cp.deepcopy(layers)
            all_layers.append(l)

        self.all_block = nn.ModuleList([])
        for i in range(sr_block_num):
            block = nn.Sequential(*all_layers[i])
            self.all_block.append(block)
        
    def forward(self, dc, con_upp, con_low):

        out = dc

        for i in range(self.sr_block_num):
            out = self.all_block[i](out)
            out = proj(out, con_upp, con_low)

        return out


# RDSRモデル
class RDSR(nn.Module):

    def __init__(self, sr_block_num):
        super(RDSR, self).__init__()

        self.sr_block_num = sr_block_num

        layers = []
        layers.append(nn.Conv2d( 1, 64, kernel_size=5, stride=1, padding=2, bias=True, padding_mode='replicate'))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=True, padding_mode='replicate'))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(32,  1, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate'))

        all_layers = []
        all_layers.append(layers)
        self.sr_block = nn.Sequential(*layers)

        
    def forward(self, dc, con_upp, con_low):

        out = dc

        for _ in range(self.sr_block_num):
            out = self.sr_block(out)
            out = proj(out, con_upp, con_low)

        return out


# フォルダの画像を読んでデータセットを作るクラス
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
 

# 凸射影
def proj(out_batch, con_upp, con_low):

    # projection 
    out_batch = out_batch * 255
    out_batch = torchjpeg.batch_dct(out_batch)
    out_batch = torch.minimum(out_batch, con_upp)
    out_batch = torch.maximum(out_batch, con_low)
    out_batch = torchjpeg.batch_idct(out_batch)
    out_batch = out_batch / 255
    return out_batch


# tensor(batch)から画像(numpy配列)に
def batch_to_ndarray(batch):
 
    ndarray = batch.cpu().detach().numpy()
    ndarray = np.squeeze(ndarray)
    ndarray = ndarray * 255
    ndarray = np.round(ndarray)
    ndarray = np.maximum(ndarray, 0)
    ndarray = np.minimum(ndarray, 255)
    ndarray = np.array(ndarray, np.uint8)
    return ndarray
 

# エントロピーなどを取得
def get_bps_bpp_aos_psnr(qtz_ind_jpg, jpg_batch, rec_batch):

    qtz_ind_jpg = qtz_ind_jpg.cpu().detach().numpy()
    qtz_ind_jpg = np.squeeze(qtz_ind_jpg)
    h, w = qtz_ind_jpg.shape

    ind_rec = torchjpeg.batch_dct(rec_batch * 255)
    ind_rec = ind_rec.cpu().detach().numpy()
    ind_rec = np.squeeze(ind_rec)

    # original sign
    org_sgn = np.sign(qtz_ind_jpg)
    org_sgn[::8, ::8] = 0

    # recovered sign
    rec_sgn = np.sign(ind_rec)
    rec_sgn[np.where(rec_sgn == 0)] = 1
    rec_sgn[::8, ::8] = 0
    rec_sgn[np.where(qtz_ind_jpg == 0)] = 0

    # bit plane to be transmitted
    bit_pln = org_sgn * rec_sgn

    # probability of residual
    num_pos = np.count_nonzero(bit_pln == +1)
    num_neg = np.count_nonzero(bit_pln == -1)
    num = num_pos + num_neg

    # 交流成分がない場合エラーで終了する
    if num == 0:
        print('error')
        exit(1)

    prb_pos = num_pos / num
    prb_neg = num_neg / num

    # residual bits per significant index
    bps = -(prb_pos * np.log2(prb_pos) + prb_neg * np.log2(prb_neg))
    
    # bits per pixel
    bpp = bps * num / (h * w)

    # accuracy of sign 
    aos = prb_pos * 100

    # psnr
    jpg_ndarray = batch_to_ndarray(jpg_batch)
    rec_ndarray = batch_to_ndarray(rec_batch)
    psnr = cv2.PSNR(jpg_ndarray, rec_ndarray)
    
    return bps, bpp, aos, psnr, rec_ndarray


# トレーニングを行う関数(引数：計算機, ネットワーク, ロス, 最適化, エポック, 入力, 教師画像)
def training(device, net, criterion, optimizer, epochs, input_loader, org_set):
 
    print('train start')

    for epoch in range(1, epochs+1):

        # 
        running_loss = 0.0

        # input_batch:dc image
        for counter, input_batch in enumerate(tqdm(input_loader), 1):

            input_batch = input_batch.to(device)

            # org_setはdataset class
            # torch.stackで最初の軸で結合
            org_list = []
            for b in range(batch_size):
                org_list.append(org_set.__getitem__(batch_size*(counter-1)+b))
            org_batch = torch.stack(org_list)
            org_batch = org_batch.to(device)
            
            # DCT coefficient
            qtz_ind_jpg = torchjpeg.compress_coefficients(org_batch, train_QF, table="luma")
            qtz_ind_our = torch.abs(qtz_ind_jpg)
            deg_cff_our = torchjpeg.dequantize_at_quality(qtz_ind_our, train_QF, table="luma")

            con_upp = deg_cff_our.clone()
            con_low = -deg_cff_our.clone()
            # 直流の符号は正
            con_low[:, :, ::8, ::8] = -con_low[:, :, ::8, ::8]

            optimizer.zero_grad()
            output = net(input_batch, con_upp, con_low)
            
            loss = criterion(org_batch, output)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / counter
        print('epoch:{0:d}, loss:{1:f}'.format(epoch, avg_loss))

    print('train finished')
    torch.save(net.state_dict(), model_save_dir + '/' + model_name + '.pth')


# QF 
train_QF = 50
test_QF = 50

# Recursiveにするかどうか
rec_flag = 1

# 学習かテストか(True:train, False:test)
train_flag = 0

# 学習時の各種パラメータ
lr_opt = 0.0002
EPOCHS = 50
batch_size = 10

sr_block_num = 1 # 反復回数
 
# uint8 -> float32, [0,255] -> [0,1], [batch, C, H, W]
transform = transforms.Compose([transforms.ToTensor()])
 
# GPUかCPUを指定
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
# RDSRのネットワークを作成
if rec_flag:
    model_name = 'RDSR'
    net = RDSR(sr_block_num)
# FDSRのネットワークを作成
else:
    model_name = 'FDSR'
    net = FDSR(sr_block_num)

# モデル保存用のフォルダを作成
model_save_dir = 'model'
os.makedirs(model_save_dir, exist_ok=True)
 
# netをGPUに送る
net = net.to(device)


# train mode
if train_flag:
 
    # 教師データ(原画像)を読み込む
    org_set = ImageFolder('train/org', transform)

    # 入力(直流画像)を読み込む
    input_set = ImageFolder('train/dc/' + str(train_QF), transform)

    # 入力をバッチごとに取り出せるようにDataLoaderで変換
    input_loader = DataLoader(input_set, batch_size=batch_size, shuffle=False)

    # ネットワークをトレーニングモードにする
    net.train()

    # ロス関数      
    criterion = torch.nn.MSELoss()

    # 最適化と学習率の設定
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_opt)

    # 学習開始
    training(device, net, criterion, optimizer, EPOCHS, input_loader, org_set)
 
 
# test mode
else:

    # tensorの勾配計算を行わないように設定
    with torch.no_grad():
        
        # 読み込むモデルを表示
        print('model:', model_name)

        # モデルの重みのパス
        model_path = model_save_dir + '/' + model_name + '.pth'

        # 学習済みモデルがなかったら強制終了
        if not os.path.isfile(model_path):
            print('model not found')
            exit(1)

        # ネットワークをテストモードにする
        net.eval()

        # ネットワークの重みを読み込む
        net.load_state_dict(torch.load(model_path, map_location=device))

        # 復元画像60枚を入れるフォルダを作成
        test_rec_dir = 'rec/test_QF=' + str(test_QF)
        os.makedirs(test_rec_dir, exist_ok=True)
        
        # 測定値の変数の初期化
        bpp_sum = 0
        bps_sum = 0
        aos_sum = 0
        psnr_sum = 0
        time_sum = 0

        for img_num in tqdm(range(1, 61)):

            # 原画像の読み込み
            org_gray = cv2.imread('test/%02d.png' % (img_num), cv2.IMREAD_GRAYSCALE)
            h, w = org_gray.shape

            # 原画像をバッチにしてGPUに送信
            org_tensor = transform(org_gray)
            org_batch = org_tensor.unsqueeze(0).to(device)

            # 量子化係数の取得
            qtz_ind_jpg = torchjpeg.compress_coefficients(org_batch, test_QF, table="luma")
            
            # 絶対値の取得
            qtz_ind_our = torch.abs(qtz_ind_jpg)
            
            # 逆量子化
            deg_cff_our = torchjpeg.dequantize_at_quality(qtz_ind_our, test_QF, table="luma")

            # 凸射影用の係数を用意
            con_upp = deg_cff_our.clone()
            con_low = -deg_cff_our.clone()
            con_low[:, :, ::8, ::8] = -con_low[:, :, ::8, ::8]

            # JPEGの取得
            jpg_batch = torchjpeg.decompress_coefficients(qtz_ind_jpg, test_QF, table="luma")

            # 入力のDCの取得
            qtz_ind_dc = torch.zeros((1, 1, h, w), dtype=torch.float32)
            qtz_ind_dc[:, :, ::8, ::8] = qtz_ind_jpg[:, :, ::8, ::8]
            input_batch = torchjpeg.decompress_coefficients(qtz_ind_dc, test_QF, table="luma")
            input_batch = input_batch.to(device)

            # 推論と実行時間の計測
            start_time = time.time()
            rec_batch = net(input_batch, con_upp, con_low)
            elapsed_time = time.time() - start_time

            # エントロピーなどを測定
            bps, bpp, aos, psnr, rec_img = get_bps_bpp_aos_psnr(qtz_ind_jpg, jpg_batch, rec_batch)
            
            # 復元画像を保存
            cv2.imwrite(test_rec_dir + '/%02d.png' % img_num, rec_img)

            # 測定値を加算
            bps_sum += bps
            bpp_sum += bpp
            aos_sum += aos
            psnr_sum += psnr
            time_sum += elapsed_time 

        # 測定値の平均を表示
        print('Avg. bps = {0:6.4f} [bit]'.format(bps_sum/60))
        print('Avg. bpp = {0:6.4f} [bpp]'.format(bpp_sum/60))
        print('Avg. accuracy of sign = {0:3.2f} [%]'.format(aos_sum/60))   
        print('Avg. PSNR = {0:3.2f} [dB]'.format(psnr_sum/60))
        print('Avg. time = {0:6.4f} [s]'.format(time_sum/60))
        


