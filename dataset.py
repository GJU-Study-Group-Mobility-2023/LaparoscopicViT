import os 
import cv2

def process_frame(frame_number, tools_present, video_number):
    # Create a folder name based on tools present in the frame
    folder_name = '-'.join(sorted(tools_present))
    
    # If the folder does not exist, create it
    folder_path = os.path.join(output_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Open the video file
    video_path = os.path.join(videos_folder, f"video{video_number:02d}.mp4")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found - {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    # Set the frame position based on frame number
    frame_position = int(frame_number)  # Assuming 25 fps

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_position >= total_frames:
        print(f"Error: Frame {frame_number} exceeds the total frames in video{video_number:02d} ({total_frames} frames)")
        cap.release()
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
    
    # Read the frame from the video
    ret, frame = cap.read()
    
    if ret:
        # Construct the new image filename
        new_image_name = f"video{video_number:02d}_frame{frame_number:04d}.jpg"
        
        # Save the frame as a jpg image in the new folder
        new_image_path = os.path.join(folder_path, new_image_name)
        cv2.imwrite(new_image_path, frame)
        
        print(f"Saved frame {frame_number} from video{video_number:02d} in {folder_name}")
    else:
        print(f"Error reading frame {frame_number} from video{video_number:02d}")

    # Release the video capture object
    cap.release()

def process_txt_file(file_path, video_number):
    with open(file_path, 'r') as file:
        # Read the header to get the tool names
        header = file.readline().strip().split('\t')[1:]

        # Read each line and process the frame
        for line in file:
            data = line.strip().split('\t')
            frame_number = int(data[0])
            tools_present = [header[i] for i in range(len(header)) if int(data[i + 1]) == 1]

            # If there are tools present, process the frame
            if tools_present:
                process_frame(frame_number, tools_present, video_number)

def videos_to_ImageFolder(annotations_folder, videos_folder, output_folder):
    for video_number in range(1, 81):  # Assuming there are 80 videos
        txt_file_path = os.path.join(annotations_folder, f"video{video_number:02d}-tool.txt")
        process_txt_file(txt_file_path, video_number)
