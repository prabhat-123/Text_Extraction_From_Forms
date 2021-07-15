import cv2
import numpy as np
import pytesseract
import os
import pickle
from tqdm import tqdm



pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

class extract_roi:
    def __init__(self,extraction_details,img_path):
        self.extraction_details = extraction_details
        self.img_path = img_path

    # Detecting Contours and Selecting COntours with area > 5000 and area < 200000
    def detect_contours(self):
        query_image = cv2.imread(self.img_path)
        gray_scale_image = cv2.cvtColor(query_image,cv2.COLOR_BGR2GRAY)
        canny_edge = cv2.Canny(gray_scale_image,5,10)
        contours,hierarchy = cv2.findContours(canny_edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        coordinates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000 and area < 200000:
                x,y,w,h = cv2.boundingRect(contour)
                coordinates.append([(x,y),(x+w,y+h)])
        return [query_image,coordinates]

    # Filtering the coordinates matching with the reference keys.
    def filter_rectangular_bbx_coordinates(self,coordinates,query_image):
        self.coordinates = coordinates
        self.query_image = query_image
        numbers = ('1.','2.','3.','4.','5.','6.','7.')
        expandable_coordinates = []
        filtered_coordinates = []
        non_filtered_coordinates = []
        clean_texts = []
        for (i,item) in enumerate(tqdm(self.coordinates)):
            (x1,y1) = item[0]
            (x2,y2) = item[1]
            roi_image = query_image[y1:y2,x1:x2]
            text = pytesseract.image_to_string(roi_image)
            for key in self.extraction_details:
                clean_text = text.strip()
                if clean_text.startswith(numbers):
                    clean_text = clean_text.replace(clean_text[0:2],'')
                    clean_text = clean_text.strip()
                if clean_text.startswith(key):
                    clean_text = clean_text.replace(key,'')
                    clean_text = clean_text.strip()
                    clean_texts.append(clean_text)
                    non_filtered_coordinates.append([(x1,y1),(x2,y2)])
                    if clean_text == '':
                        expandable_coordinates.append([(x1,y1),(x2,y2)])
                    else:
                        filtered_coordinates.append([(x1,y1),(x2,y2)])
        return [expandable_coordinates,non_filtered_coordinates,filtered_coordinates,clean_texts]

    # Calculating the remaining coordinates by removing the unwanted coordinates
    def calculate_remaining_coordinates(self,coordinates,non_filtered_coordinates):
        self.coordinates = coordinates
        self.non_filtered_coordinates = non_filtered_coordinates
        remaining_coordinates = [[i[0],i[1]] for i in coordinates if i not in self.non_filtered_coordinates]
        return remaining_coordinates

    # Expand the rectangles to get the desired region of interest. Adding two smaller rectangles
    # to get the bigger rectangle which covers the region of interest
    def calculate_expanded_coordinates(self,expandable_coordinates,remaining_coordinates):
        self.expandable_coordinates = expandable_coordinates
        self.remaining_coordinates = remaining_coordinates
        expanded_coordinates = []
        for coord in self.expandable_coordinates:
            (x1,y1) = coord[0]
            (x2,y2) = coord[1]
            for item in self.remaining_coordinates:
                (x3,y3) = item[0]
                (x4,y4) = item[1]
                if x1 == x3 and x2 == x4 and y4 > y1:
                    expanded_coordinates.append([(x1,y1),(x4,y4)])
        return expanded_coordinates

    # Removing some unwanted expanded coordinates to remove the duplicate coordinates
    # covering the region of interest
    def remove_unwanted_expanded_coordinates(self,expanded_coordinates):
        self.expanded_coordinates = expanded_coordinates
        query_image = self.detect_contours()[0]
        clean_texts = self.filter_rectangular_bbx_coordinates(coordinates=self.expanded_coordinates,query_image=query_image)[3]
        for i in range(len(clean_texts)):
            text_present = any(key in clean_texts[i] for key in self.extraction_details)
            if text_present== True:
                self.expanded_coordinates.pop(i)
        return self.expanded_coordinates
    
    # Extracting the coordinates of the region of interest by combining filtered 
    # coordinates and expanded coordinates.
    def extract_roi_coordinates(self,new_expanded_coordinates,filtered_coordinates):
        self.new_expanded_coordinates = new_expanded_coordinates
        self.filtered_coordinates = filtered_coordinates
        roi_list = self.filtered_coordinates + self.new_expanded_coordinates
        return roi_list

    # Since 'ETD of POL' and 'ETA of POL' both falls in same region of interest. So,
    # splitting the larger coordinate into two coordinates to get two separate rois.
    def splitting_larger_coordinates(self,roi_coordinates):
        self.roi_coordinates = roi_coordinates
        query_image = self.detect_contours()[0]
        new_roi_coordinates = []
        numbers = ('1.','2.','3.','4.','5.','6.','7.')
        for (i,item) in enumerate(tqdm(self.roi_coordinates)):
            (x1,y1) = item[0]
            (x2,y2) = item[1]
            roi_image = query_image[y1:y2,x1:x2]
            text = pytesseract.image_to_string(roi_image)
            clean_text = text.strip()
            if clean_text.startswith('ETD of POL'):
                gray_scale_image = cv2.cvtColor(roi_image,cv2.COLOR_BGR2GRAY)
                canny_edge = cv2.Canny(gray_scale_image,5,20)
                contours, hierarchy = cv2.findContours(canny_edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
                line_coordinates = []
                for contour in contours:  
                    x,y,w,h = cv2.boundingRect(contour)
                    if h > 70:
                        line_coordinates.append([(x,y), (x+w,y+h)])
                [(x3,y3),(x4,y4)] = [line_coordinates[0][0],line_coordinates[0][1]]
                [(x5,y5),(x6,y6)] = [(x1,y1),(x1+x4,y1+y4)]
                new_roi_coordinates.append([(x5,y5),(x6,y6)])
                [(x7,y7),(x8,y8)] = [(x1+x4,y1),(x2,y2)]
                new_roi_coordinates.append([(x7,y7),(x8,y8)])
            else:
                new_roi_coordinates.append([item[0],item[1]])
        return new_roi_coordinates

    # Extracting the coordinates of hbl_num and mbl_num so that it can be referenced 
    # when extracting their respective codes.. Since 'SAAC code' is common reference key 
    # so by knowing coordinates of hbl_num and mbl_num, we can check whether the 'SAAC'
    #  code lies below hbl_num or mbl_num.
    def extracting_hblandmbl_coordinates(self,updated_roi_coordinates):
        self.updated_roi_coordinates = updated_roi_coordinates
        query_image = self.detect_contours()[0]
        numbers = ('1.','2.','3.','4.','5.','6.','7.')
        hbl_coordinate = []
        mbl_coordinate = []
        for (i,item) in enumerate(tqdm(self.updated_roi_coordinates)):
            (x1,y1) = item[0]
            (x2,y2) = item[1]
            roi_image = query_image[y1:y2,x1:x2]
            text = pytesseract.image_to_string(roi_image)
            clean_text = text.strip()
            if clean_text.startswith('HB/L# used in AMS filling'):
                hbl_coordinate.append([(x1,y1),(x2,y2)])
            elif clean_text.startswith('Regular or Straight B/L'):
                mbl_coordinate.append([(x1,y1),(x2,y2)])
        return [hbl_coordinate,mbl_coordinate]

    # Pickling the region of interest, coordinates of hbl_num and mbl_num so that
    # it can be used during inference for faster response.
    def pickling_roi_list(self,updated_roi_coordinates,hbl_coordinate,mbl_coordinate):
        self.updated_roi_coordinates = updated_roi_coordinates
        self.hbl_coordinate = hbl_coordinate
        self.mbl_coordinate = mbl_coordinate
        f1 = open('roi_list.pkl', "wb")
        pickle.dump(self.updated_roi_coordinates,f1)
        f1.close()
        f2 = open('mbl_list.pkl','wb')
        pickle.dump(self.mbl_coordinate,f2)
        f2.close()
        f3 = open('hbl_list.pkl','wb')
        pickle.dump(self.hbl_coordinate,f3)
        f3.close()









    
