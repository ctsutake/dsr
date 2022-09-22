-----------------------------------------------------------------------------

     Compressing Sign Information in 
     DCT-based Image Coding via Deep Sign Retrieval         
 
-----------------------------------------------------------------------------

Written by  : Kei Suzuki
Affiliation : Nagoya University
E-mail      : tsutake.chihiro.c3@f.mail.nagoya-u.ac.jp
Created     : Sep. 2022

-----------------------------------------------------------------------------
    Contents
-----------------------------------------------------------------------------

model/RDSR.pty : RDSR model (trained)
main.py        : Main algorithm file
torchjpeg.py   : TorchJPEG[1]

[1] M. Ehrlich et al., "Quantization Guided JPEG Artifact Correction,"
in Proc. European Conference on Computer Vision (ECCV), 2020.

-----------------------------------------------------------------------------
    Usage
-----------------------------------------------------------------------------

-- Training

1) Download and extract the following zip.

train.zip
https://drive.google.com/file/d/1NO-cLuQ99gpNLj1ENEz4NyKyOH_zClzV/view?usp=sharing

2) Set train_flg = 1 (line 213).

3) Change train QF (line 209), learning rate (line 216), epochs (line 219), and K (line 225).

4) Run 'python3 main.py'.

-- Test

1) Download and extract the following zip.

test.zip
https://drive.google.com/file/d/1b7K3VCsbaPSZeLeg0P6ZLOhyeBYbHoGh/view?usp=sharing

2) Set train_flg = 0 (line 213).

3) Change test QF (line 210).

4) Run 'python3 main.py'.

-----------------------------------------------------------------------------
    Feedback
-----------------------------------------------------------------------------

If you have any questions, please contact me.
