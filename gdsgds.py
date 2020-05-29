__author__ = "Gds"
import os , face_recognition,cv2,numpy as np
#~~~~~~~~~~~        视频抓取测试      ~~~~~~~~~~~~~~~
"""
构建videocaptrue对象
"""
video_capture =  cv2.VideoCapture(0)
"""
加载人物图片 numpy.ndarray类型
"""
obama_image = face_recognition.load_image_file("高尚/高尚1.jpg")
obama_image1 = face_recognition.load_image_file("1/33.jpg")
"""
获取每个图像文件中每个面部的面部编码
由于每个图像中可能有多个人脸，所以返回一个编码列表。
但是事先知道每个图像只有一个人脸，每个图像中的第一个编码，取索引0。
"""
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
obama_face_encoding1 = face_recognition.face_encodings(obama_image1)[0]


known_face_encodings = [
    obama_face_encoding,
obama_face_encoding1

]
known_face_names = [
    "gaodashang",
    "chenglong"

]

# Initialize some variables
# 初始化一些变量
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    #抓取每一帧视频
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.4)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()












# images = os.listdir("2")
#
# images2  = os.listdir("1")
# image_to_be_matched = face_recognition.load_image_file('1/33.jpg')
# #
# # image_to_be_matched = face_recognition.load_image_file('2/640.jpg')
# # ~~~~~~~~~~~     第一版         ~~~~~~~~~~~~~~~
#
# face_locations = face_recognition.face_locations(image_to_be_matched)
# face_landmarks_list = face_recognition.face_landmarks(image_to_be_matched)
# print(type(image_to_be_matched))
# biden_encoding = face_recognition.face_encodings(image_to_be_matched)[0]
# print(type(face_locations))
# for i in images:
#
#     img = face_recognition.load_image_file("2/" + i)
#     unknown_encoding = face_recognition.face_encodings(img)
#
#     if unknown_encoding:
#         unknown_encoding = unknown_encoding[0]
#         results = face_recognition.compare_faces([biden_encoding], unknown_encoding,tolerance=0.45)
#         if results[0] == True:
#             print("匹配成功")
#         else:
#             print("匹配失败")
#







#~~~~~~~~~~~         网版         ~~~~~~~~~~~~~~

# image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0]
# count = 0
# count2 = 0
# for image in images:
#
#     # load the image
#     try:
#         current_image = face_recognition.load_image_file("2/" + image)
#
#         # encode the loaded image into a feature vector
#
#         current_image_encoded = face_recognition.face_encodings(current_image)[0]
#
#
#         # match your image with the image and check if it matches
#
#         result = face_recognition.compare_faces(
#
#             [image_to_be_matched_encoded], current_image_encoded)
#
#         # check if it was a match
#         print(result)
#         if result[0] == True:
#             count2+=1
#             print("匹配成功: " + image)
#
#         else:
#             count += 1
#             print("匹配失败: " + image)
#     except :
#         count += 1
#         print("匹配失败: " + image)
#
# print("共%d个 ，失败%d个， 成功%d个"%(len(images),count,count2))
