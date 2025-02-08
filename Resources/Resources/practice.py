from cvzone.ClassificationModule import Classifier
import cv2

cap = cv2.VideoCapture(0)
classifier = Classifier(r'C:\Users\DELL\Desktop\Resources\Resources\Model\keras_model.h5',r'C:\Users\DELL\Desktop\Resources\Resources\Model\labels.txt')

while True:
    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    prediction, index = classifier.getPrediction(img)
    print(prediction, index)
    cv2.imshow("Image", img)
    cv2.waitKey(1)



