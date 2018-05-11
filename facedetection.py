import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

image = cv2.imread("sample.jpeg");

# cv2.imshow("Sample Image",image);
# cv2.waitKey(0);
# cv2.destroyAllWindows();

# storing the returned pixels where the face is detected 


gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray Image", gray_image);
# cv2.waitKey(0);

faces = face_cascade.detectMultiScale(gray_image,
scaleFactor=1.05,
minNeighbors=5);


print(faces);
print(type(faces));

# Drawing the rectangle around the face of the image
for x,y,w,h in faces:
	image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2);


cv2.imshow("Face Detected Image",image);
cv2.waitKey(0);
cv2.destroyAllWindows();