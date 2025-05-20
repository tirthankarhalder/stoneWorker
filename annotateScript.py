import sys
from datetime import datetime, timedelta
from tkinter import Tk, simpledialog

import cv2
import pandas as pd

video_path = "video_2024-12-02_15-03-19_nano1.mp4"
start_time_str = video_path.split("_")[1] + " " + video_path.split("_")[2]
start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H-%M-%S")


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video file: {video_path}")
    sys.exit(-1)


fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Processing {video_path} | FPS: {fps}, Total Frames: {total_frames}")

annotations = []
frame_number = 0

# Initialize a basic GUI for user input
root = Tk()
root.withdraw()  # Hide the main Tkinter window

while frame_number < total_frames:
    start = simpledialog.askstring(
        "Annotate Segment",
        f"Video: {video_path}\n"
        f"Enter start time (HH.MM.SS) for current annotation:"
    )

    end = simpledialog.askstring(
        "Annotate Segment",
        f"Video: {video_path}\n"
        f"Enter end time (HH.MM.SS) for current annotation:"
    )

    # Convert HH:MM:SS to integers
    start_t = start
    end_t = end
    start = start.split(".")
    end = end.split(".")
    for i in range(len(start)):
        start[i] = int(start[i])
    for i in range(len(end)):
        end[i] = int(end[i])
    start = int(start[0] * 3600 + start[1] * 60 + start[2])
    end = int(end[0] * 3600 + end[1] * 60 + end[2])

    chunk_size = (end - start + 1) * 30

    # Read frames in chunks
    frames = []
    for _ in range(chunk_size):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_number += 1

    if not frames:
        break

    # cv2.imshow("Annotate Chunk (Press Q to Quit)", frames[-1])
    # Prompt user to annotate the segment
    start_frame = frame_number - len(frames)
    end_frame = frame_number - 1

    label = simpledialog.askstring(
        "Annotate Segment",
        f"Video: {video_path}\n"
        f"Frames: {start_frame} - {end_frame}\n"
        f"Enter activity:"
    )

    if label:
        for i in range(len(frames)):
            current_frame = start_frame + i
            current_time = start_time + timedelta(seconds=current_frame / fps)
            timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            annotations.append({
                "Video": video_path,
                "Frame": current_frame,
                "Timestamp": timestamp,
                "Start Time": start_t,
                "End Time": end_t,
                "Start Frame": start,
                "End Frame": end,
                "Activity": label
            })
    else:
        print(f"Skipped segment {start_frame}-{end_frame}.")

    print(f"Number of frames processed: {len(frames)}")
    isLast = simpledialog.askinteger(
        "Annotate Segment",
        f"Video: {video_path}\n"
        f"Was that the last annotation?\n"
        f"Enter 0 for 'YES' and 1 for 'NO':",
        minvalue=0,
        maxvalue=1
    )

    if isLast == 0:
        print("Annotation stopped by user.")
        break

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     print("Annotation stopped by user.")
    #     break

cap.release()
cv2.destroyAllWindows()
root.destroy()
# print(annotations)


df = pd.DataFrame(annotations)
file_name = f"{video_path[:-4]}.csv"
df.to_csv(path_or_buf=f"./annotations/{file_name}", index=False)
print(f"Saved data to {file_name}")
