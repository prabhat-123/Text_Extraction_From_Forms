from main_program import extract_roi
image_path = 'job_assignment_img.jpg'

reference_keys = ['Seller Name & Address', 'Buyer Name & Address(Importer of Record)',
'Container','POL','POD','ETA of POD','ETD of POL','Vessel/Voyage',
'Regular or Straight B/L','SCAC code','HB/L# used in AMS filling','Type of movement']

obj = extract_roi(extraction_details=reference_keys,img_path=image_path)
[query_image,coordinates] = obj.detect_contours()
print("Extracting rectangular coordinates from Images")
[expandable_coordinates,non_filtered_coordinates,filtered_coordinates,clean_texts]=obj.filter_rectangular_bbx_coordinates(coordinates=coordinates,query_image=query_image)
remaining_coordinates = obj.calculate_remaining_coordinates(coordinates=coordinates,non_filtered_coordinates=non_filtered_coordinates)
expanded_coordinates = obj.calculate_expanded_coordinates(expandable_coordinates=expandable_coordinates,remaining_coordinates=remaining_coordinates)
print("Removing unwanted expanded coordinates")
new_expanded_coordinates = obj.remove_unwanted_expanded_coordinates(expanded_coordinates=expanded_coordinates)
roi_coordinates = obj.extract_roi_coordinates(new_expanded_coordinates=expanded_coordinates,filtered_coordinates=filtered_coordinates)
print("Updating ROI coordinates")
updated_roi_coordinates = obj.splitting_larger_coordinates(roi_coordinates=roi_coordinates)
print("Extracting HBL And MBL coordinates")
[hbl_coordinate,mbl_coordinate] = obj.extracting_hblandmbl_coordinates(updated_roi_coordinates=updated_roi_coordinates)
obj.pickling_roi_list(updated_roi_coordinates=updated_roi_coordinates,hbl_coordinate=hbl_coordinate,mbl_coordinate=mbl_coordinate)