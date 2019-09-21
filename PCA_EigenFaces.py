import numpy as np
import scipy as sp
from PIL import Image
from resizeimage import resizeimage
import glob
import os

directory_path = 'D:\\Courses\\Fall 19\\ELEG 815 Statistical Learning\\Homeworks\\Homework3_Files(1)\\'
A = np.empty((25,38400))
def Preprocess_Data(directory_path):
    i = 0
    for img_path in glob.glob(directory_path+'/*.*'):
        with Image.open(img_path) as image:
            gray_resizedImage = resizeimage.resize_contain(image,[240,160]).convert('L')               # Source - Stackoverflow
            pixels = list(gray_resizedImage.getdata())
            A[i] = pixels
            i += 1

def Calculate_Average_Faces(A,j):                                               # j = row number in matrix A
    Sum_List = [sum(x) for x in zip(A[j],A[j+1],A[j+2],A[j+3],A[j+4])]          # use zip when adding 2 or more iterables
    n = 5
    avg_vector = [(Sum_List[i])/n for i in range(len(Sum_List))]
    return avg_vector

def Reconstruct_Image_from_Pixels(avg_list,height,width):
    A_Reconstructed = np.empty((160,240))
    cntr = 0
    for i in range(height):
        for j in range(width):
            A_Reconstructed[i][j] = avg_list[cntr]
            cntr += 1
    return A_Reconstructed

Preprocess_Data(directory_path)
Anne = Calculate_Average_Faces(A,0)
Benjamin = Calculate_Average_Faces(A,5)
Keanu = Calculate_Average_Faces(A,10)
Markle = Calculate_Average_Faces(A,15)
Ryan = Calculate_Average_Faces(A,20)

Avg_Array_Anne = Reconstruct_Image_from_Pixels(Anne,160,240)
Avg_Array_Benjamin = Reconstruct_Image_from_Pixels(Benjamin,160,240)
Avg_Array_Keanu = Reconstruct_Image_from_Pixels(Keanu,160,240)
Avg_Array_Markle = Reconstruct_Image_from_Pixels(Markle,160,240)
Avg_Array_Ryan = Reconstruct_Image_from_Pixels(Ryan,160,240)

Avg_Image_Anne = Image.fromarray(Avg_Array_Anne)
Avg_Image_Benjamin = Image.fromarray(Avg_Array_Benjamin)
Avg_Image_Keanu = Image.fromarray(Avg_Array_Keanu)
Avg_Image_Markle = Image.fromarray(Avg_Array_Markle)
Avg_Image_Ryan = Image.fromarray(Avg_Array_Ryan)

Avg_Image_Anne.show()
Avg_Image_Benjamin.show()
Avg_Image_Keanu.show()
Avg_Image_Markle.show()
Avg_Image_Ryan.show()





