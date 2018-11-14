# medSynthesis with TensorFlow (we also provide an improved version in pytorch, you can access it at https://github.com/ginobilinie/medSynthesisV1. Since the pytorch version is recently updated, all the APIs are to the latest and the network is more advanced, I suggest your use the pytorch version.)

This project is for medical image synthesis with generative adversarial networks (GAN), such as, synthesize CT from MRI, 7T from 3T, high does PET from low dose PET.

Currently, we have uploaded a 2D/3D GAN in this repository (2D is in the root folder, and 3D version is in the folder of '3dversion'). Detailed information can be found in our paper: 

<a  href="https://link.springer.com/chapter/10.1007/978-3-319-66179-7_48">Medical Image Synthesis with Context-Aware Generative Adversarial Networks</a>

If it is helpful for you, please cite our paper:

@inproceedings{nie2017medical,
  title={Medical image synthesis with context-aware generative adversarial networks},
  author={Nie, Dong and Trullo, Roger and Lian, Jun and Petitjean, Caroline and Ruan, Su and Wang, Qian and Shen, Dinggang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={417--425},
  year={2017},
  organization={Springer}
}

# How to run the tensorflow code
The main entrance for the code is main.py

I suppose you have installed:    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensorflow (>=0.12.1)
     <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;simpleITK 
     <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;numpy

Steps to run the code (since the code is implemented as early as 2016, part of the codes are deprecated, you can refer to <a href='https://github.com/ginobilinie/medSynthesisV1'>our pytorch version</a> for new implementations):
1. use readMedImg4CaffeCropNie4SingleS.py to extract patches (as limited annotated data can be acquired in medical image fields, we usually use patch as the training unit), and save as hdf5 format.
2. modify the g_model.py if you want to do some changes to the architecture of the generator
3. modify the d_model.py if you want to do some changes to the architecture of the discriminator
4. check the loss function in the loss_functions.py.
5. set up the hyper-parameters in the main.py
6. run the code: python main.py

# Dataset
BTW, you can download a real medical image synthesis dataset for reconstructing standard-dose PET from low-dose PET via this link:  https://www.aapm.org/GrandChallenge/LowDoseCT/

Also, there are some MRI synthesis datasets available:
http://brain-development.org/ixi-dataset/

# License
medSynthesis is released under the MIT License (refer to the LICENSE file for details).

