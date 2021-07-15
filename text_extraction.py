import pickle
import os
import numpy as np
import cv2
import pytesseract
from tqdm import tqdm
import json

file1 = open('roi_list.pkl','rb')
roi_list = pickle.load(file1)

file2 = open('mbl_list.pkl','rb')
mbl_coordinate = pickle.load(file2)

file3 = open('hbl_list.pkl','rb')
hbl_coordinate = pickle.load(file3)

reference_keys = ['Seller Name & Address', 'Buyer Name & Address(Importer of Record)',
'Container','POL','POD','ETA of POD','ETD of POL','Vessel/Voyage','Vessel/Voyage',
'Regular or Straight B/L','SCAC code','HB/L# used in AMS filling','SCAC code','Type of movement']

output_keys = ['seller','buyer','container','pol','pod','eta','etd','vessel_name','voyage_num',
'mbl_num','mbl_scac','hbl_num','hbl_scac','type_of_movement']


query_img = cv2.imread('job_assignment_img.jpg')

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

output_dictionary = {}
numbers = ('1.','2.','3.','4.','5.','6.','7.')
print("Extracting texts from Region Of Interest")
for (i,item) in enumerate(tqdm(roi_list)):
    (x1,y1) = item[0]
    (x2,y2) = item[1]
    roi_image = query_img[y1:y2,x1:x2]
    text = pytesseract.image_to_string(roi_image)
    for key in reference_keys:
        clean_text = text.strip()
        if clean_text.startswith(numbers):
            clean_text = clean_text.replace(text[0:2],'')
            clean_text = clean_text.strip()
        if clean_text.startswith(key):
            for i in range(len(output_keys)):
                if key == reference_keys[i]:
                    output_key = output_keys[i]
                    clean_text =  clean_text.replace(key,'')
                    clean_text = clean_text.strip()
                    output_value = clean_text.replace('\n',' ')
                    if output_key == 'vessel_name':
                        output_value = output_value.split(' ')[0:3]
                        output_value = ' '.join(map(str, output_value))
                        output_dictionary[output_key] = output_value
                    elif output_key == 'voyage_num':
                        output_value = output_value.split(' ')[-1]
                        output_dictionary[output_key] = output_value
                    elif key == 'SCAC code':
                        [(x3,y3),(x4,y4)] = [hbl_coordinate[0][0],hbl_coordinate[0][1]]
                        [(x5,y5),(x6,y6)] = [mbl_coordinate[0][0],mbl_coordinate[0][1]]
                        if x1 == x3 and x2 == x4:
                            output_key = 'hbl_scac'
                            output_dictionary[output_key] = output_value
                        elif x1 == x5 and x2 == x6:
                            output_key = 'mbl_scac'
                            output_dictionary[output_key] = output_value
                    else:
                        output_dictionary[output_key] = output_value

with open("output.json", "w") as outfile:  
    json.dump(output_dictionary, outfile) 