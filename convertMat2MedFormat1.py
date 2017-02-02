    
'''
Target: Crop patches for kinds of medical images, such as hdr, nii, mha, mhd, raw and so on, and store them as hdf5 files
for single-scale patches
Created on Oct. 20, 2016
Author: Dong Nie 
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np

d1=64
d2=64
d3=5
dFA=[d1,d2,d3] # size of patches of input data
dSeg=[48,48,1] # size of pathes of label data
step1=8
step2=8
step3=5
step=[step1,step2,step3]
    
'''
Actually, we donot need it any more,  this is useful to generate hdf5 database
'''
def cropCubic(matFA,matSeg,fileID,d,step,rate):
  
    eps=1e-5
    rate1=1.0/2
    rate2=1.0/4
    [row,col,leng]=matFA.shape
    cubicCnt=0
    estNum=40000
    trainFA=np.zeros([estNum,1, dFA[0],dFA[1],dFA[2]])
    trainSeg=np.zeros([estNum,1,dSeg[0],dSeg[1],dSeg[2]])
    print 'trainFA shape, ',trainFA.shape
    #to padding for input
    margin1=(dFA[0]-dSeg[0])/2
    margin2=(dFA[1]-dSeg[1])/2
    margin3=(dFA[2]-dSeg[2])/2
    cubicCnt=0
    marginD=[margin1,margin2,margin3]
    print 'matFA shape is ',matFA.shape
    matFAOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    print 'matFAOut shape is ',matFAOut.shape
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA
    matSegOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg
    #for mageFA, enlarge it by padding
    if margin1!=0:
        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]
    #for matseg, enlarge it by padding
    if margin1!=0:
        matSegOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matSegOut[row+marginD[0]:matSegOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg[matSeg.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matSegOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matSeg[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matSegOut[marginD[0]:row+marginD[0],col+marginD[1]:matSegOut.shape[1],marginD[2]:leng+marginD[2]]=matSeg[:,matSeg.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matSeg[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matSegOut.shape[2]]=matSeg[:,:,matSeg.shape[2]-1:leng-marginD[2]-1:-1]
        
    dsfactor = rate
    
    for i in range(0,row-dSeg[0],step[0]):
        for j in range(0,col-dSeg[1],step[1]):
            for k in range(0,leng-dSeg[2],step[2]):
                volSeg=matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                #if np.sum(volSeg)<eps:
                #    continue
                cubicCnt=cubicCnt+1
                #index at scale 1
            
                
                volFA=matFAOut[i:i+dFA[0],j:j+dFA[1],k:k+dFA[2]]
              
                trainFA[cubicCnt,0,:,:,:]=volFA #32*32*32
            
                trainSeg[cubicCnt,0,:,:,:]=volSeg#24*24*24


    trainFA=trainFA[0:cubicCnt,:,:,:,:]
  
    trainSeg=trainSeg[0:cubicCnt,:,:,:,:]

    with h5py.File('./train64to48_%s.h5'%fileID,'w') as f:
        f['data3T']=trainFA
        f['data7T']=trainSeg
     
    with open('./train64to48_list.txt','a') as f:
        f.write('./train64to48_%s.h5\n'%fileID)
    return cubicCnt
    	
def main():
    path='/shenlab/lab_stor3/dongnie/3T7T-Data/'
    saveto='/shenlab/lab_stor3/dongnie/3T7T-Data/'
   
    ids=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    for ind in ids:
        datafilename='S%d/3t.hdr'%ind #provide a sample name of your filename of data here
        datafn=os.path.join(path,datafilename)
        labelfilename='S%d/7t.hdr'%ind  # provide a sample name of your filename of ground truth here
        labelfn=os.path.join(path,labelfilename)
        imgOrg=sitk.ReadImage(datafn)
        mrimg=sitk.GetArrayFromImage(imgOrg)
       	mu=np.mean(mrimg)
       	maxV, minV=np.percentile(mrimg, [99 ,25])
       	#mrimg=mrimg
       	mrimg=(mrimg-mu)/(maxV-minV)

        labelOrg=sitk.ReadImage(labelfn)
        labelimg=sitk.GetArrayFromImage(labelOrg) 
       	mu=np.mean(labelimg)
       	maxV, minV=np.percentile(labelimg, [99 ,25])
      	#labelimg=labelimg
       	labelimg=(labelimg-mu)/(maxV-minV)
        #you can do what you want here for for your label img
        #imgOrg=sitk.ReadImage(gtfn)
        #gtMat=sitk.GetArrayFromImage(imgOrg)
        prefn='s%d_3t.nii.gz'%ind
        preVol=sitk.GetImageFromArray(mrimg)
        sitk.WriteImage(preVol,prefn)
        outfn='s%d_7t.nii.gz'%ind
        preVol=sitk.GetImageFromArray(labelimg)
        sitk.WriteImage(preVol,outfn)
 
        fileID='%d'%ind
        rate=1
        #cubicCnt=cropCubic(mrimg,labelimg,fileID,dFA,step,rate)
        #print '# of patches is ', cubicCnt
    
if __name__ == '__main__':     
    main()
