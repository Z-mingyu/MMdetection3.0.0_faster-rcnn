import cv2
import os

# 视频的帧数为30，那么60s的视频一共有1800帧，选择60张照片，则选择每隔30帧选一张

path_video =os.listdir("/home/zmy/workspace/Data/thermometer")
save_pictures = "/home/zmy/workspace/Image/"

count = 721
for num in range(len(path_video)):
    # 视频数量一共是9个
    path_cv = "/home/zmy/workspace/Data/thermometer/"+path_video[num]
    vc = cv2.VideoCapture(path_cv)

    i = 0
    while vc.isOpened():
        rval, img = vc.read()

        frame_count = vc.get(cv2.CAP_PROP_FRAME_COUNT)  # 视频文件的帧数
        frame_fps = vc.get(cv2.CAP_PROP_FPS)  # 视频文件的帧率

        if i == frame_count:
            break
        else:
            i = i + 1
            if i % int(frame_count / 60) == 0:  # 选取60照片
                count = count + 1
                # 图片命名及保存路径
                filename = "{:0>5}.jpg".format(count)
                # 调整图像大小
                img = cv2.resize(img, (960, 540))
                cv2.imwrite(save_pictures + filename, img)

    print("第{}个视频截取图片完毕！".format(num))

    # 释放资源
    vc.release()
    cv2.destroyAllWindows()
