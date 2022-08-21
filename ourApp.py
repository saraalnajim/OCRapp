#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install streamlit
#!python -m pip install paddlepaddle
#!pip install paddleocr


# In[1]:


from paddleocr import PaddleOCR, draw_ocr  #OCR
import streamlit as st  #Web App
import matplotlib.pyplot as plt
from collections import namedtuple
import argparse
import imutils
import cv2
import numpy as np
import re
import pandas as pd
from PIL import Image
import time
import multiprocessing as mp
import os.path as path

#title
st.title("Extract Nutrients from Images")

#subtitle
st.markdown("Image Alignment & Optical Character Recognition  - Using `Paddleocr`, `streamlit`, `OpenCV`")

#image uploader
image = st.file_uploader(label = "Upload your image here",type=['png'])


@st.cache
def alignment(im2):
    refFilename=r"shared code\image alignment fils\good_form.png" 
    #Read reference image
    im1=cv2.imread(refFilename,cv2.IMREAD_COLOR)
    im1=cv2.cvtColor(im1,cv2.COLOR_BGR2RGB)
    
    #Convert images to grayscale
    im1_gray=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    #Detect ORB features and compute descriptors.
    MAX_NUM_FEATURES=20000
    GOOD_MATCH_PERCENT=0.05
    orb=cv2.ORB_create(MAX_NUM_FEATURES)
    keypoints1,descriptors1=orb.detectAndCompute(im1_gray,None)
    keypoints2,descriptors2=orb.detectAndCompute(im2_gray,None)
    
    #Match features.
    matcher=cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches=matcher.match(descriptors1,descriptors2,None)
    #Sort matches by score
    matches= sorted(matches, key=lambda x: x.distance, reverse=False)
    #Remove not so good matches
    numGoodMatches=int(len(matches)*GOOD_MATCH_PERCENT)
    matches=matches[:numGoodMatches]

    #Draw top matches
    im_matches=cv2.drawMatches(im1,keypoints1,im2,keypoints2,matches,None)

    #Extract location of good matches
    points1=np.zeros((len(matches),2),dtype=np.float32)
    points2=np.zeros((len(matches),2),dtype=np.float32)
    for i,match in enumerate(matches):
        points1[i,:]=keypoints1[match.queryIdx].pt
        points2[i,:]=keypoints2[match.trainIdx].pt
    #Find homography
    H, Mask = cv2.estimateAffine2D(points1, points2, cv2.RANSAC)
    H = np.vstack((H, [0, 0, 1]))
    h,mask=cv2.findHomography(points2,points1,cv2.RANSAC)

    #Use homography to warp image
    height,width,channels=im1.shape
    im2_reg=cv2.warpPerspective(im2,h,(width, height))
    
    return im2_reg

def ocr(im2):
    # create a named tuple which we can use to create locations of the
    # input document which we wish to OCR
    OCRLocation = namedtuple("OCRLocation", ["id", "bbox","filter_keywords"])
    # define the locations of each area of the document we wish to OCR
    OCR_LOCATIONS = [
        OCRLocation("RecipeName", (810, 927, 396, 81),["|","[","]"]),
        OCRLocation("nfacts", (423, 1257, 804, 1176),[""]),
        OCRLocation("allergen astatement", (306, 2544, 1002, 348),[""]),
                ]
    num_list= ["Total Fat","Saturated Fat","Trans Fat","Polyunsaturated Fat","Monounsaturated Fat","Cholesterol","Sodium",
           "Total Carbohydrate","Dietary Fiber","Total Sugars","Protein"]

    allergen_astatement_list=["wheat" , "milk", "egg","strawberry","strawberries","banana","mango","nuts","tree nuts",
                          "soy","shellfish","fish","dairy","legumes"]
    #read image
    image = alignment(im2)

    #import and initialize PaddleOCR model
    from paddleocr import PaddleOCR, draw_ocr
    ocr_model = PaddleOCR(lang='en')



    # initialize a dictionary to store our final OCR results
    results = {
    "RecipeName":"",
    "servings per container":"" ,
    "Serving size":"",
    "Calories":"",
    "Total Fat":"",
    "Saturated Fat":"",
    "Trans Fat":"",
    "Polyunsaturated Fat":"",
    "Monounsaturated Fat":"",
    "Cholesterol":"",
    "Sodium":"",
    "Total Carbohydrate":"",
    "Dietary Fiber":"",
    "Total Sugars":"",
    "Added Sugars":"",
    "Protein":"",
    "allergen astatement":""
          }
    # loop over the locations of the document we are going to OCR

    for loc in OCR_LOCATIONS:
        text=""
        # extract the OCR ROI from the aligned image
        (x, y, w, h) = loc.bbox
        roi = image[y:y + h, x:x + w]
    
        ##Paddle OCR
        readText = ocr_model.ocr(roi)
    
    
        for i in readText:
            text+=i[1][0]+"\n"
    
    
    
        #######fillter text 
    
        # record the value of Recipe Name in the dictionary
        if loc.id == "RecipeNamet":
            results["RecipeName"]= text
            
        
        #record the value of allergens in the dictionary
        if loc.id == "allergen astatement": 
            text = text.lower()
            #Initialize allergens variable
            allergens=""
            #loop over the alrgens word if find any return it 
            for i in allergen_astatement_list:
                #will return the word if find it as a list, and the length of this 
                #list is how many times this word appears
                #try print(len(allergen_list)) to understand
                allergen_list = re.findall(i,text)
                #compine the list cell if the length not equal zero
                allergens+= "".join([x+", " for x in allergen_list if len(x)!= 0 ])
            #replace last occurrence of " ," with dot
            results["allergen astatement"]= allergens[::-1].replace(" ,",".", 1)[::-1]
        
        
        if loc.id == "nfacts":
            # break the text into lines and loop over them
            for line in text.split("\n"):
                # find the value of Serving size by retrieving the line with "("
                if "(" in line:
                    results["Serving size"]= line[1:-1]  
            #take 2 indexes before the word "servings per container"
            #will return index of first letter in the statement
            index = text.find("servings per container")
            results["servings per container"]=text[index-2:index]    
            
            
         
        
            #take 4 indexes before the word "Added Sugars"
            index = text.find("Added Sugars")
            results["Added Sugars"]= text[index-4:index]
        
            #take 10 indexes after last letter of "Amount per serving"  
            index = text.find("Amount per serving")
            i=index+len("Amount per serving")
            #then return numbers only
            results["Calories"]= re.sub("[^0-9^.]", "", text[i:i+10])
        
            #loop over the rest of macros and take 5 indexes after each word
            for word in num_list:
                line=""
                index = text.find(word)+len(word)
                line = text[index:index+8]
                results[word]= line
    d={}
    for key, value in results.items():
        #clean dictionary values
        #take frist line of the string
        value= value.partition('\n')[0]
        #remove whitespaces
        value= value.strip()
        value= value.replace("O", "0")
        d[key] = [value]
    df2 = pd.DataFrame(d)
    return df2
        



if image is not None:

    #فوق نحط اسم الفورم وتحت اسم الصوره 
    refFilename=r"good_form.png" 
    #imFilename=r"shared code\image alignment fils\Nfacts_photo1.png"
    im2 = Image.open(image).convert('RGB')

    #Read reference image
    im1=cv2.imread(refFilename,cv2.IMREAD_COLOR)
    im1=cv2.cvtColor(im1,cv2.COLOR_BGR2RGB)

    #Read image to be aligned
    #im2=cv2.imread(image,cv2.IMREAD_COLOR)
    im2 = np.array(im2)
    #im2=cv2.cvtColor(im2,cv2.COLOR_BGR2RGB)
    st.image(im2) #display image
    start = time.time()

    with st.spinner("🤖 AI is at Work! "):
        df=ocr(im2)
        st.dataframe(df)
        
        # get the end time
        end = time.time()
        # get the execution time
        elapsed_time = end - start
        st.write('Execution time:', round(elapsed_time), 'seconds')
        df.to_csv(r"C:\Users\maria\OneDrive - IMAM ABDULRAHMAN BIN FAISAL UNIVERSITY\Training\AI\results.csv", index=False)
        
        st.success('CSV File is saved to your computer')
    st.balloons()
else:
    st.write("Upload an Image")

st.caption("Made with ❤️ by Mariam & Sara")


# In[ ]:


get_ipython().system('streamlit run ourApp.py')


# In[ ]:





# In[ ]:




