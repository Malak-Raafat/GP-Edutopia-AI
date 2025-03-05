import cv2
import os
from ultralytics import YOLO
import imagehash
from PIL import Image

# Load the trained YOLO model
model = YOLO(r"D:\data\colab with pretrained and augmentation.pt")  # Update with your trained model path

# Define paths
video_path = r"D:\data\videos\Breaking.Bad.S05E14.EgyDead.CoM.mp4"  # Update with your video file path
output_folder = r"D:\data\cropped_objects_local_colab_pretrained_aug"
os.makedirs(output_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_interval = fps * 10  # Extract frame every 5 seconds

frame_count = 0
object_count = 0  # To count and name cropped objects
confidence_threshold = 0  # Set confidence threshold
min_size_threshold = 0  # Minimum area threshold (e.g., 5000 pixels)
min_resolution_threshold = 200  # Minimum width & height threshold
margin = 40  # Margin around the bounding box (in pixels)
hash_threshold = 9  # Hash difference threshold for similarity (adjust as needed)

# Store hashes of saved images grouped by class
saved_hashes = {}  # Format: {class_name: set_of_hashes}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Resize the frame to a width of 1024 pixels, maintaining aspect ratio
    height, width = frame.shape[:2]
    new_width = 640
    new_height = int((new_width / width) * height)
    frame = cv2.resize(frame, (new_width, new_height))

    if frame_count % frame_interval == 0:  # Process every 5 seconds
        results = model(frame)  # Run YOLO on the frame

        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):  
                conf = float(conf)  # Convert confidence to float
                class_id = int(cls)  # Convert class index to integer
                class_name = model.names[class_id]  # Get class name from model

                # 🔴 Skip saving if the class is "legend"
                if class_name.lower() == "legend":
                    print(f"⏭️ Skipping 'legend' object (Conf: {conf:.2f})")
                    continue  

                conf_str = f"{conf:.2f}"  # Format confidence score to 2 decimal places

                if conf >= confidence_threshold:  # Apply confidence threshold
                    x1, y1, x2, y2 = map(int, box)
                    width = x2 - x1  # Width of the bounding box
                    height = y2 - y1  # Height of the bounding box

                    # 🔴 Skip objects that are too small (both width & height < min_resolution_threshold)
                    if width < min_resolution_threshold and height < min_resolution_threshold:
                        print(f"⏭️ Skipping small object ({class_name}) [{width}x{height}]")
                        continue  

                    # Add margin to the bounding box coordinates (only for objects that passed the size check)
                    x1 = max(0, x1 - margin)  # Ensure x1 doesn't go below 0
                    y1 = max(0, y1 - margin)  # Ensure y1 doesn't go below 0
                    x2 = min(frame.shape[1], x2 + margin)  # Ensure x2 doesn't exceed frame width
                    y2 = min(frame.shape[0], y2 + margin)  # Ensure y2 doesn't exceed frame height

                    width_with_margin = x2 - x1  # New width after adding margin
                    height_with_margin = y2 - y1  # New height after adding margin
                    area_with_margin = width_with_margin * height_with_margin  # New area after adding margin

                    if area_with_margin >= min_size_threshold:  # Apply size threshold
                        cropped_object = frame[y1:y2, x1:x2]  # Crop detected object with margin

                        if cropped_object.size > 0:  # Ensure valid crop
                            # Convert the cropped image to PIL format for hashing
                            pil_image = Image.fromarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB))
                            image_hash = imagehash.phash(pil_image)  # Compute perceptual hash

                            # Initialize saved_hashes for the class if not already done
                            if class_name not in saved_hashes:
                                saved_hashes[class_name] = set()

                            # Check if the image is similar to any previously saved image of the same class
                            is_similar = False
                            for saved_hash in saved_hashes[class_name]:
                                if image_hash - saved_hash < hash_threshold:  # Compare hashes
                                    is_similar = True
                                    break

                            if is_similar:
                                print(f"⏭️ Skipping similar object ({class_name})")
                                continue

                            # Save the image and its hash
                            object_count += 1
                            resolution = f"{width_with_margin}x{height_with_margin}"  # Get resolution of cropped image
                            save_path = os.path.join(output_folder, f"{class_name}_{conf_str}_{resolution}.jpg")
                            cv2.imwrite(save_path, cropped_object)  # Save cropped object
                            saved_hashes[class_name].add(image_hash)  # Store the hash for this class

                            # Display image with confidence
                            display_frame = cropped_object.copy()
                            label = f"{class_name}: {conf_str} ({resolution})"
                            cv2.putText(display_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                        1, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.imshow("Detected Object", display_frame)
                            cv2.waitKey(500)  # Show each image for 500ms

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"✅ Process completed! {object_count} objects saved in {output_folder} (excluding 'legend', small objects, and similar images)")