import cv2

def detectFace(img):
    filename = img.split(".")[0] # 取得檔案名稱(不添加副檔名)
    img = cv2.imread(img) # 讀取圖檔
    # 顯示原始圖片
    cv2.imshow("Original_Image",img)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 透過轉換函式轉為灰階影像
    # 顯示灰階圖片
    cv2.imshow("Gray_Image",grayImg)
    color = (0, 255, 0)  # 定義框的顏色
    
    # OpenCV 人臉識別分類器
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # 調用偵測識別人臉函式
    faceRects = face_classifier.detectMultiScale(
        grayImg, scaleFactor=1.2, minNeighbors=1, minSize=(32, 32))
    
    # 大於 0 則檢測到人臉
    if len(faceRects):  
        # 框出每一張人臉
        for faceRect in faceRects: 
            x, y, w, h = faceRect
            cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
    # 顯示結果圖片
    cv2.imshow("Image",img)
    # 將結果圖片輸出
    cv2.imwrite(filename + "_face.jpg", img)
    
detectFace('human.jpg')
cv2.waitKey()