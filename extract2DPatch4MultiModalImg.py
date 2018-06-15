    
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
dSeg=[64,64,1] # size of pathes of label data
step1=8
step2=8
step3=1
step=[step1,step2,step3]
    

class ScanFile(object):   
    def __init__(self,directory,prefix=None,postfix=None):  
        self.directory=directory  
        self.prefix=prefix  
        self.postfix=postfix  
          
    def scan_files(self):    
        files_list=[]    
            
        for dirpath,dirnames,filenames in os.walk(self.directory):   
            ''''' 
            dirpath is a string, the path to the directory.   
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..'). 
            filenames is a list of the names of the non-directory files in dirpath. 
            '''  
            for special_file in filenames:    
                if self.postfix:  
                    if  special_file.endswith(self.postfix):    
                        files_list.append(os.path.join(dirpath,special_file))    
                elif self.prefix:    
                    if special_file.startswith(self.prefix):  
                        files_list.append(os.path.join(dirpath,special_file))    
                else:    
                    files_list.append(os.path.join(dirpath,special_file))    
                                  
        return files_list    
      
    def scan_subdir(self):  
        subdir_list=[]  
        for dirpath,dirnames,files in os.walk(self.directory):  
            subdir_list.append(dirpath)  
        return subdir_list      
    

    
'''
Actually, we donot need it any more,  this is useful to generate hdf5 database
'''
def extractPatch4OneSubject(matFA, matMR, matSeg, matMask, fileID ,d, step, rate):
  
    eps=1e-2
    rate1=1.0/2
    rate2=1.0/4
    [row,col,leng]=matFA.shape
    cubicCnt=0
    estNum=40000
    trainFA=np.zeros([estNum,1, dFA[0],dFA[1],dFA[2]])
    trainSeg=np.zeros([estNum,1,dSeg[0],dSeg[1],dSeg[2]])
    trainMR=np.zeros([estNum,1,dSeg[0],dSeg[1],dSeg[2]])

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
    
    matMROut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    print 'matMROut shape is ',matMROut.shape
    matMROut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matMR
    
    matSegOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg
 
 
    matMaskOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    matMaskOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matMask
       
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
        
      #for matMR, enlarge it by padding
    if margin1!=0:
        matMROut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matMR[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matMROut[row+marginD[0]:matMROut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matMR[matMR.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matMROut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matMR[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matMROut[marginD[0]:row+marginD[0],col+marginD[1]:matMROut.shape[1],marginD[2]:leng+marginD[2]]=matMR[:,matMR.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matMROut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matMR[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matMROut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matMROut.shape[2]]=matMR[:,:,matMR.shape[2]-1:leng-marginD[2]-1:-1]    
    
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
  
      #for matseg, enlarge it by padding
    if margin1!=0:
        matMaskOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matMask[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matMaskOut[row+marginD[0]:matMaskOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matMask[matMask.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matMaskOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matMask[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matMaskOut[marginD[0]:row+marginD[0],col+marginD[1]:matMaskOut.shape[1],marginD[2]:leng+marginD[2]]=matMask[:,matMask.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matMaskOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matMask[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matMaskOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matMaskOut.shape[2]]=matMask[:,:,matMask.shape[2]-1:leng-marginD[2]-1:-1]
              
    dsfactor = rate
    
    for i in range(0,row-dSeg[0],step[0]):
        for j in range(0,col-dSeg[1],step[1]):
            for k in range(0,leng-dSeg[2],step[2]):
                volMask = matMaskOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                if np.sum(volMask)<eps:
                    continue
                cubicCnt = cubicCnt+1
                #index at scale 1
                volSeg = matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                volFA = matFAOut[i:i+dFA[0],j:j+dFA[1],k:k+dFA[2]]
                
                volMR = matMROut[i:i+dFA[0],j:j+dFA[1],k:k+dFA[2]]

                trainFA[cubicCnt,0,:,:,:] = volFA #32*32*32
            
                trainSeg[cubicCnt,0,:,:,:] = volSeg#24*24*24


    trainFA = trainFA[0:cubicCnt,:,:,:,:]
    trainMR = trainMR[0:cubicCnt,:,:,:,:]  
    trainSeg = trainSeg[0:cubicCnt,:,:,:,:]

    with h5py.File('./trainPETCT_64_%s.h5'%fileID,'w') as f:
        f['dataLPET'] = trainFA
        f['dataCT'] = trainMR        
        f['dataHPET'] = trainSeg
     
    with open('./trainPETCT_64_list.txt','a') as f:
        f.write('./trainPETCT_64_%s.h5\n'%fileID)
    return cubicCnt
        
def main():
    path = '/home/niedong/Data4LowDosePET/'
    scan = ScanFile(path, postfix = '60s_suv.nii.gz')  
    filenames = scan.scan_files()  
   
    maxLPET = 149.366742
    maxPercentLPET =
    minLPET = 0.00055037
    meanLPET = 0.27593288
    stdLPET = 0.75747500
    
    # for s-pet
    maxSPET = 156.675962
    maxPercentSPET =
    minSPET = 0.00055037
    meanSPET = 0.284224789
    stdSPET = 0.7642257
    
    # for rsCT
    maxCT = 27279
    maxPercentCT = 
    minCT = -1023
    meanCT = -601.1929
    stdCT = 475.034
    
    for filename in filenames:
         
        print 'low dose filename: ', filename
        
        lpet_fn = filename
        ct_fn = filename.replace('60s_suv','rsCT')  
        spet_fn = filename.replace('60s_suv','120s_suv')
        
        imgOrg = sitk.ReadImage(lpet_fn)
        mrimg = sitk.GetArrayFromImage(imgOrg)
        
        maskimg = mrimg
        
#         maxV, minV = np.percentile(mrimg, [99.5 ,1])
#         print 'maxV is: ',np.ndarray.max(mrimg)
#         mrimg[np.where(mrimg>maxV)] = maxV
#         print 'maxV is: ',np.ndarray.max(mrimg)
#         mu=np.mean(mrimg) # we should have a fixed std and mean
#         std = np.std(mrimg) 
#         mrnp = (mrimg - mu)/std
#         print 'maxV,',np.ndarray.max(mrnp),' minV, ',np.ndarray.min(mrnp)

        matLPET = (mrimg - meanLPET)/(stdLPET)
        print 'maxV,',np.ndarray.max(matLPET),' minV, ',np.ndarray.min(matLPET)



        imgOrg1 = sitk.ReadImage(ct_fn)
        mrimg1 = sitk.GetArrayFromImage(imgOrg1)
        
#         maxV1, minV1 = np.percentile(mrimg1, [99.5 ,1])
#         print 'maxV1 is: ',np.ndarray.max(mrimg1)
#         mrimg1[np.where(mrimg1>maxV1)] = maxV1
#         print 'maxV1 is: ',np.ndarray.max(mrimg1)
#         mu1 = np.mean(mrimg1) # we should have a fixed std and mean
#         std1 = np.std(mrimg1) 
#         mrnp1 = (mrimg1 - mu1)/std1
#         print 'maxV1,',np.ndarray.max(mrnp1),' minV, ',np.ndarray.min(mrnp1)

        mrimg1[np.where(mrimg1>maxPercentCT)] = maxPercentCT
        matCT = (mrimg1 - meanCT)/stdCT
        print 'maxV,',np.ndarray.max(matCT),' minV, ',np.ndarray.min(matCT)
      

        labelOrg = sitk.ReadImage(spet_fn)
        labelimg = sitk.GetArrayFromImage(labelOrg) 
        
#         maxVal = np.amax(labelimg)
#         minVal = np.amin(labelimg)
#         print 'maxV is: ', maxVal, ' minVal is: ', minVal
#         mu=np.mean(labelimg) # we should have a fixed std and mean
#         std = np.std(labelimg) 
#         
#         labelimg = (labelimg - minVal)/(maxVal - minVal)
# 
#         print 'maxV,',np.ndarray.max(labelimg),' minV, ',np.ndarray.min(labelimg)
        #you can do what you want here for for your label img
        
        matSPET = (mrimg1 - minSPET)/(maxPercentCT-minSPET)
        print 'maxV,',np.ndarray.max(matSPET),' minV, ',np.ndarray.min(matSPET)
                
        
        fileID = '%d'%ind
        rate = 1
        cubicCnt = extractPatch4OneSubject(mrimg1, mrimg1, labelimg, maskimg, fileID,dSeg,step,rate)
        print '# of patches is ', cubicCnt
    
if __name__ == '__main__':     
    main()
