import cv2

img = cv2.imread('./image/001.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
## 閾值分割
ret,thresh = cv2.threshold(gray,200,255,1)

## 對二值圖像執行膨脹操作
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5, 5))     
dilated = cv2.dilate(thresh,kernel)

## 輪廓提取，cv2.RETR_TREE表示建立層級結構
""" image,  """
contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

## 提取小方格，其父輪廓都為0號輪廓
boxes = []
for i in range(len(hierarchy[0])):
    if hierarchy[0][i][3] == 0:
        boxes.append(hierarchy[0][i])

## 提取數字，其父輪廓都存在子輪廓 
number_boxes = []
for j in range(len(boxes)):
    if boxes[j][2] != -1:
        #number_boxes.append(boxes[j])
        x,y,w,h = cv2.boundingRect(contours[boxes[j][2]])
        number_boxes.append([x,y,w,h])
        img = cv2.rectangle(img,(x-1,y-1),(x+w+1,y+h+1),(0,0,255),2)

cv2.namedWindow("img", cv2.WINDOW_NORMAL); 
cv2.imshow("img", img)
cv2.waitKey(0) 