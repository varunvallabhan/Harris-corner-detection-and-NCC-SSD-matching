#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:25:48 2018

@author: byakuya
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 02:27:30 2018

@author: byakuya
"""

import cv2
import numpy as np
image1 = cv2.imread('/Users/byakuya/work/computer vision/homework4/HW4Pics/pair1/1.jpg')
image2 = cv2.imread('/Users/byakuya/work/computer vision/homework4/HW4Pics/pair1/2.jpg')
image3 = cv2.imread('/Users/byakuya/work/computer vision/homework4/HW4Pics/pair2/truck1.jpeg')
image4 = cv2.imread('/Users/byakuya/work/computer vision/homework4/HW4Pics/pair2/truck2.jpeg')
image5 = cv2.imread('/Users/byakuya/work/computer vision/homework4/HW4Pics/pair1/t11.jpeg')
image6 = cv2.imread('/Users/byakuya/work/computer vision/homework4/HW4Pics/pair1/t12.jpeg')
image7 = cv2.imread('/Users/byakuya/work/computer vision/homework4/HW4Pics/pair1/t3.jpeg')
image8 = cv2.imread('/Users/byakuya/work/computer vision/homework4/HW4Pics/pair1/t4.jpeg')


def Harris(img,gaussian,gk,sigm,sobel_k,window,name):


    out=img.copy()
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(gaussian,gaussian),gk)
    corner=[]
    
    #calculating the differenciation
    z=img.shape[0]
    w=img.shape[1]
    x=np.array
    #dy, dx = np.gradient(gray)                          tried gradient function got similar if now lower accuracy
    dx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_k)           #used sobel derivative instead 
    dy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_k)
    dx2=dx**2
    dy2=dy**2                                           #calculatinga the parameter matrix
    dxy=dx*dy
                                                #window/2 
    for x in range(sigm,z-sigm):
        for y in range(sigm,w-sigm):         #iterating over the image to calculate the sum of differences
            eledx2 = dx2[x-sigm:x+sigm, y-sigm:y+sigm]
            eledy2 = dy2[x-sigm:x+sigm, y-sigm:y+sigm]
            eledxy = dxy[x-sigm:x+sigm, y-sigm:y+sigm]
            sumdx2= eledx2.sum()
            sumdy2= eledy2.sum()
            sumdxy= eledxy.sum()
            if dx[x,y]%dy[x,y]!=0  and dy[x,y]>0 and dx[x,y]>0:
               # r=(sumdx2*sumdy2-sumdxy**2)/(sumdx2+sumdy2)**2
                r=(sumdx2*sumdy2-sumdxy**2)-0.04*(sumdx2+sumdy2)**2
                if r>10000 :
                    nei = gray[x-window:x+window, y-window:y+window]
                    nei.astype('uint64')
                    #des= nei.sum()
                    corner.append([x,y,r,nei])
            
    
    corner.sort(key=lambda x:x[2], reverse=True) 
    res=[]
    corner=corner[:1000]
    #fi=[]
    #corner.sort(key=lambda x:x[0], reverse=True)
    for a in range(3,len(corner)-3):
        if (corner[a+1][0] not in range(corner[a][0]-5, corner[a][0]+5)):
            #if (corner[a+1][1] not in range(corner[a][1]-5, corner[a][1]+5)):    
            res.append(corner[a])
    #res.sort(key=lambda x:x[0], reverse=True)
    #for a in range(0,len(res)-1):
     #   if (res[a+1][0] not in range(res[a][0]-1, res[a][0]+1)):
     #       fi.append(res[a])
    
    for a in range(2,len(res)):
        cv2.circle(out,(res[a][1],res[a][0]),1,(200,0,0),2)
            
    cv2.imwrite('/Users/byakuya/work/computer vision/homework4/HW4Pics/pair1/'+name+'.jpg',out)
    
    return res

def ssd(image1,image2,cor1,cor2,window,thresh,name):
    
    t=[]
    
    #print cor1[156][3][1][0]
    #if len(cor1)>= len(cor2):
    for a in range(0,len(cor1)):
        flag=0
        ssd=[]
        if cor1[a][3].size == (2*window)**2 :
            for z in range(0,len(cor2)):
                s=0
                if cor2[z][3].size ==(2*window)**2:
                    dif=cor1[a][3]-cor2[z][3]
                    s=(np.sum(np.multiply(dif,dif)))      #ssd
                        
                    ssd.append(s)
                    if s<thresh and flag==0:           #filtering recurring matche
                        t.append([a,z])
                        flag=1
    print min(ssd)
            
            #creating new image and mapping
    width=image1.shape[0]+image2.shape[0]
    height=max(image1.shape[1],image2.shape[1])
    image=np.zeros([width,height,3],dtype='uint8')
    for i in range(0,width):
        for j in range(0,height):
            if i<((image.shape[0]/2)-1):
                    # image[i][j]=image1[i]][j]
                    
                image[i][j]= image1[i][j]
                x=i
            else:
                    
                image[i][j]=image2[i-x-2][j]
        
            
    for a in range(0,len(t)):
                    
        cv2.circle(image,(cor1[t[a][0]][1],cor1[t[a][0]][0]),3,(200,0,0),2)
        cv2.circle(image,(cor2[t[a][1]][1],(cor2[t[a][1]][0]+(width/2))),3,(200,0,0),2)
        cv2.line(image,(cor1[t[a][0]][1],cor1[t[a][0]][0]),(cor2[t[a][1]][1],cor2[t[a][1]][0]+width/2),(255,0,0),2)
    cv2.imwrite('/Users/byakuya/work/computer vision/homework4/HW4Pics/pair1/'+name+'.jpg',image)
    return t
     
def Ncc(image1,image2,cor1,cor2,window,thresh,name):
    t=[]
    ncc=[]
    test=[]
    for a in range(0,len(cor1)):
        flag=0
        if cor1[a][3].size == (2*window)**2 :
            for z in range(0,len(cor2)):
                if cor2[z][3].size == (2*window)**2 :
                    m1=np.mean(cor1[a][3])
                    m2=np.mean(cor2[z][3])
                    t1=cor1[a][3]-m1
                    t2=cor2[z][3]-m2
                    num=np.sum(np.multiply(t1,t2))
                    den=np.sqrt(np.multiply(np.sum(np.multiply(t1,t1)),np.sum(np.multiply(t2,t2))))
                    s=np.true_divide(num,den)
                    ncc.append(s)
                    if s>thresh and flag==0 and z not in test :
                        test.append(z)
                        t.append([a,z])
                        flag=1 
            
    print max(ncc)
            #creating new image and mapping
    width=image1.shape[0]+image2.shape[0]
    height=max(image1.shape[1],image2.shape[1])
    image=np.zeros([width,height,3],dtype='uint8')
    for i in range(0,width):
        for j in range(0,height):
            if i<((image.shape[0]/2)-1):
                    # image[i][j]=image1[i]][j]
                    
                image[i][j]= image1[i][j]
                x=i
            else:
                    
                image[i][j]=image2[i-x-2][j]
        
            
    for a in range(0,len(t)):
                    
        cv2.circle(image,(cor1[t[a][0]][1],cor1[t[a][0]][0]),1,(200,0,0),2)
        cv2.circle(image,(cor2[t[a][1]][1],(cor2[t[a][1]][0]+(width/2))),1,(200,0,0),2)
        cv2.line(image,(cor1[t[a][0]][1],cor1[t[a][0]][0]),(cor2[t[a][1]][1],cor2[t[a][1]][0]+width/2),(255,0,0),1)
    cv2.imwrite('/Users/byakuya/work/computer vision/homework4/HW4Pics/pair1/'+name+'.jpg',image)

cor1=Harris(image7,3,3,3,5,6,'sigman31')
cor2=Harris(image8,3,3,3,5,6,'sigman32')
cor3=Harris(image7,3,3,5,5,6,'sigman51')
cor4=Harris(image8,3,3,5,5,6,'sigman52')
cor5=Harris(image7,3,3,7,5,6,'sigman71')
cor6=Harris(image8,3,3,7,5,6,'sigman72')
cor7=Harris(image7,3,3,9,5,6,'sigman91')
cor8=Harris(image8,3,3,9,5,6,'sigman92')
#sd=ssd(image5,image6,cor1,cor2,6,1000,'imt1')
Ncc(image7,image8,cor1,cor2,6,0.99,'sigman312')
Ncc(image7,image8,cor3,cor4,6,0.99,'sigman512')
Ncc(image7,image8,cor5,cor6,6,0.99,'sigman712')
Ncc(image7,image8,cor7,cor8,6,0.99,'sigman912')