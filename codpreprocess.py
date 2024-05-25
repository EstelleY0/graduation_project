import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture("/Users/estelleyoon/Downloads/project_grad/bb_1_161001_vehicle_29_215_N.mp4")

prev_boxes = None

output_image_path = "/Users/estelleyoon/Downloads/project_grad/output_images/"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # 클래스 ID가 2인 차량에 대해서만 처리
                # 감지된 객체의 경계 상자 좌표 계산
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    if boxes:
        bottom_center_y = 0
        selected_box = None
        frame_height = frame.shape[0]
        for x, y, w, h in boxes:
            box_center_y = y + h // 2
            if box_center_y > bottom_center_y:
                bottom_center_y = box_center_y
                selected_box = [x, y, w, h]

        x, y, w, h = selected_box

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        pts = np.array([[0, frame.shape[0]],  # Bottom-left corner
                        [frame.shape[1], frame.shape[0]],  # Bottom-right corner
                        [frame.shape[1]//2-100, 2*frame.shape[0] // 3]], np.int32)  # Top-center

        cv2.fillPoly(mask, [pts], (255, 255, 255))

        mask[y:y + h, max(x, 0):x + w] = 255
        result = cv2.bitwise_and(frame, frame, mask=mask)

        bottom_height = int(frame.shape[0] * 0.1)
        result[-bottom_height:] = frame[-bottom_height:]

        cv2.imwrite(output_image_path + "frame_{}.jpg".format(int(cap.get(cv2.CAP_PROP_POS_FRAMES))), result)

    elif prev_boxes is not None:
        x, y, w, h = prev_boxes

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        pts = np.array([[0, frame.shape[0]],  # Bottom-left corner
                        [frame.shape[1], frame.shape[0]],  # Bottom-right corner
                        [frame.shape[1]//2-100, 2*frame.shape[0] // 3]], np.int32)  # Top-center

        cv2.fillPoly(mask, [pts], (255, 255, 255))

        mask[y:y + h, max(x, 0):x + w] = 255

        result = cv2.bitwise_and(frame, frame, mask=mask)

        bottom_height = int(frame.shape[0] * 0.1)
        result[-bottom_height:] = frame[-bottom_height:]

        cv2.imwrite(output_image_path + "frame_{}.jpg".format(int(cap.get(cv2.CAP_PROP_POS_FRAMES))), result)

    else:
        # 결과가 없는 경우 원본 프레임 저장
        cv2.imwrite(output_image_path + "frame_{}.jpg".format(int(cap.get(cv2.CAP_PROP_POS_FRAMES))), frame)

    cv2.imshow('Result', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
