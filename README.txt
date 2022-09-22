-----------------------------------------------------------------------------

     Compressing Sign Information in 
     DCT-based Image Coding via Deep Sign Retrieval         
 
-----------------------------------------------------------------------------

Written by  : Kei Suzuki
Affiliation : Nagoya University
E-mail      : tsutake.chihiro.c3@f.mail.nagoya-u.ac.jp
Created     : Feb. 2022

-----------------------------------------------------------------------------
    Contents
-----------------------------------------------------------------------------

model/RDSR/pty : RDSR model (trained)
main.py        : Main algorithm file
torchjpeg.py   : TorchJPEG[1]

[1] M. Ehrlich et al., "Quantization Guided JPEG Artifact Correction,"
in Proc. European Conference on Computer Vision (ECCV), 2020.

-----------------------------------------------------------------------------
    Usage
-----------------------------------------------------------------------------

-- Training
1) 

-- Test
1) Change the current directory to `PK99' or `ZLXLMG16'.
2) Choose the variable `qf' in `main.m' from the range 1 to 100.
2) Running `main' generates the following images.

    -- jpg.png (decoded image)
    -- res.png (restored image)

-----------------------------------------------------------------------------
    Feedback
-----------------------------------------------------------------------------

If you have any questions, please contact me.
