#-*-coding:utf8-*-

import os
import cv2
import numpy as np

def read_data():
   
    img_rows,img_cols,img_depth=32,32,15
    X_tr=[] 
    #Reading boxing action class
    listing = os.listdir('./kth-dataset/boxing')
    for vid in listing:
        vid = './kth-dataset/boxing/'+vid
        frames = []
        cap = cv2.VideoCapture(vid)
        fps = cap.get(5)
        for k in range(15):
            ret, frame = cap.read()
            frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        inputs=np.array(frames)
        ipt=np.rollaxis(np.rollaxis(inputs,2,0),2,0)
        X_tr.append(ipt)
    print("boxing done")

    #Reading hand clapping action class
    listing2 = os.listdir('./kth-dataset/handclapping')
    for vid2 in listing2:
        vid2 = './kth-dataset/handclapping/'+vid2
        frames = []
        cap = cv2.VideoCapture(vid2)
        fps = cap.get(5)
        for k in range(15):
            ret, frame = cap.read()
            frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        input=np.array(frames)
        ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
        X_tr.append(ipt)
    print("handclapping done")

    #Reading hand waving action class
    listing3 = os.listdir('./kth-dataset/handwaving')
    for vid3 in listing3:
        vid3 = './kth-dataset/handwaving/'+vid3
        frames = []
        cap = cv2.VideoCapture(vid3)
        fps = cap.get(5)
        for k in range(15):
            ret, frame = cap.read()
            frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        input=np.array(frames)
        ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
        X_tr.append(ipt)
    print("handwaving done")

    #Reading jogging action class
    listing4 = os.listdir('./kth-dataset/jogging')
    for vid4 in listing4:
        vid4 = './kth-dataset/jogging/'+vid4
        frames = []
        cap = cv2.VideoCapture(vid4)
        fps = cap.get(5)
        for k in range(15):
            ret, frame = cap.read()
            frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        input=np.array(frames)
        ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
        X_tr.append(ipt)
    print("jogging done")

    #Reading running action class
    listing5 = os.listdir('./kth-dataset/running')
    for vid5 in listing5:
        vid5 = './kth-dataset/running/'+vid5
        frames = []
        cap = cv2.VideoCapture(vid5)
        fps = cap.get(5)
        for k in range(15):
            ret, frame = cap.read()
            frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        input=np.array(frames)
        ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
        X_tr.append(ipt)
    print("running done")

    #Reading walking action class
    listing6 = os.listdir('./kth-dataset/walking')
    for vid6 in listing6:
        vid6 = './kth-dataset/walking/'+vid6
        frames = []
        cap = cv2.VideoCapture(vid6)
        fps = cap.get(5) # cv2.cv.CV_CAP_PROP_FPS ; cv2.CAP_PROP_FPS
        for k in range(15):
            ret, frame = cap.read()
            frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        input=np.array(frames)
        ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
        X_tr.append(ipt)
    print("walking done")

    return X_tr
