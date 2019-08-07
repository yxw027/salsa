import cv2


new_framerate = 13
input = cv2.VideoCapture("../results/GNSSHardware/GVK/tracked.avi")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter("../results/GNSSHardware/GVK/tracked" + str(new_framerate) + ".avi", fourcc, new_framerate, (752, 480))

while (input.isOpened()):
    ret, frame = input.read()
    if not ret:
        break
    output.write(frame)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)


