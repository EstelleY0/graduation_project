import os
import cv2

def crop_and_resize(frame, size=224):
    # Get dimensions of the frame
    height, width, _ = frame.shape

    # Determine the size of the square crop
    min_dim = min(height, width)

    # Calculate the coordinates for cropping to center
    start_x = (width - min_dim) // 2
    start_y = (height - min_dim) // 2
    end_x = start_x + min_dim
    end_y = start_y + min_dim

    # Crop the frame to a square
    cropped_frame = frame[start_y:end_y, start_x:end_x]

    # Resize the cropped frame to the desired size
    resized_frame = cv2.resize(cropped_frame, (size, size))

    return resized_frame

def extract_frames(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each video file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4") or filename.endswith(".avi"):  # Assuming video files are in .mp4 or .avi format
            video_path = os.path.join(input_folder, filename)
            video_name = os.path.splitext(filename)[0]

            # Create a subfolder for this video
            video_output_folder = os.path.join(output_folder, video_name.replace("\\", ""))
            if not os.path.exists(video_output_folder):
                os.makedirs(video_output_folder)

            # Open the video file
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frame_count = 0

            # Extract frames
            while True:
                success, frame = video_capture.read()
                if not success:
                    break
                frame_count += 1
                if frame_count % (int(fps) // 4) == 0:  # Extract every 4th frame
                    # Crop and resize the frame
                    processed_frame = crop_and_resize(frame)

                    # Save the processed frame
                    frame_filename = os.path.join(video_output_folder, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_filename, processed_frame)

            print(f"Finished extracting frames from {filename}")

            video_capture.release()

# Example usage
input_folder = "C:/Users/skku/Desktop/New folder/video"    //modify
output_folder = "C:/Users/skku/Desktop/New folder/frames"  //modify
extract_frames(input_folder, output_folder)
