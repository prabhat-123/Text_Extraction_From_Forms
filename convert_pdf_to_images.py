from pdf2image import convert_from_path   
# Store Pdf with convert_from_path function 
images = convert_from_path('job_assignment.pdf') 
  
for img in images: 
    img.save('output.jpg', 'JPEG')