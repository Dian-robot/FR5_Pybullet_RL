import cv2

# video1=cv2.VideoCapture(4)
video2=cv2.VideoCapture(6)
# video2.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 设置宽度为2K
# video2.set(cv2.CAP_PROP_FRAME_HEIGHT, 290)
# fps1 = video1.get(cv2.CAP_PROP_FPS)
# size1 = (int(video1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps2 = video2.get(cv2.CAP_PROP_FPS)
size2 = (int(video2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

if  (not video2.isOpened()):
    print("Error: Camera cannot be opened.")
else:
    # cv2.namedWindow('1', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('1', size1[0], size1[1])  # 设置窗口大小
    cv2.namedWindow('2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('2', size2[0], size2[1])  # 设置窗口大小
    while True:
        # ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()
        if (not ret2):
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # cv2.imshow('1', frame1)
        cv2.imshow('2', frame2)
        # Exit condition
        if cv2.waitKey(1) == ord('q'):
            break

# video1.release()
cv2.destroyAllWindows()

