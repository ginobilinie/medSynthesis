# medSynthesis with Tensorflow

This project is for medical image synthesis with generative adversarial networks (GAN).

Currently, we have uploaded a 2D GAN in this repository (3D version will also be shared soon). Detailed information can be found in our paper: 

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

The main entrance for the code is main.py

I suppose you have installed:    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensorflow (>=0.12.1)
     <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;simpleITK 
     <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;numpy

Steps to run the code:
1. use readMedImg4CaffeCropNie4SingleS.py to extract patches (as limited annotated data can be acquired in medical image fields, we usually use patch as the training unit), and save as hdf5 format.
2. modify the g_model.py if you want to do some changes to the architecture of the generator
3. modify the d_model.py if you want to do some changes to the architecture of the discriminator
4. check the loss function in the loss_functions.py.
5. set up the hyper-parameters in the main.py
6. run the code: python main.py

