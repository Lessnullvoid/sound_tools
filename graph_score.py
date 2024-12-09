import cv2
import numpy as np
from pythonosc import udp_client
import argparse

#todo 
# add L loop functionality to play the score
# add R to reverse scan


def analyze_image(image, display_image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Draw keypoints on the display image with green and blue colors
    for i, kp in enumerate(keypoints):
        color = (0, 255, 0) if i % 2 == 0 else (0, 0, 0)  # Alternate between green and blue
        cv2.drawKeypoints(display_image, [kp], display_image, color, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Analyze keypoints
    contrast = gray.std()
    object_count = len(keypoints)
    sizes = [kp.size for kp in keypoints]
    
    # Calculate size metrics
    min_size = min(sizes) if sizes else 0
    avg_size = np.mean(sizes) if sizes else 0
    max_size = max(sizes) if sizes else 0
    
    # Calculate proximity as the average distance of keypoints from the center
    center_x, center_y = gray.shape[1] // 2, gray.shape[0] // 2
    proximity = np.mean([np.sqrt((kp.pt[0] - center_x) ** 2 + (kp.pt[1] - center_y) ** 2) for kp in keypoints])
    
    return contrast, object_count, sizes, proximity, min_size, avg_size, max_size

def add_info_box(image, contrast, object_count, proximity, duration, scan_data=None, scan_object_count=None):
    # Create a black rectangle in the lower right corner
    h, w = image.shape[:2]
    start_x = w - 140  # 140 pixels from right
    start_y = h - 120  # 100 pixels from bottom
    image[start_y:h, start_x:w] = (0, 0, 0)  # Black rectangle
    
    # Add text with white color
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (255, 255, 255)  # White color
    thickness = 1
    
    if scan_data is not None:
        # Calculate metrics for the scanned area
        scan_array = np.array(scan_data)
        scan_mean = np.mean(scan_array)
        scan_std = np.std(scan_array)
        scan_max = np.max(scan_array)
        
        texts = [
            f"Duration: {duration:.1f}s",
            f"Total Objects: {object_count}",  # Changed text to clarify total objects
            f"Scan Mean: {scan_mean:.1f}",
            f"Scan Max: {scan_max:.1f}",
            f"Scan Objects: {scan_object_count}"
        ]
    else:
        texts = [
            f"Duration: {duration:.1f}s",
            f"Contrast: {contrast:.1f}",
            f"Objects: {object_count}",
            f"Proximity: {proximity:.1f}"
        ]
    
    # Position and draw each line of text
    for i, text in enumerate(texts):
        y_position = start_y + 20 + (i * 20)  # 20 pixels between lines
        cv2.putText(image, text, (start_x + 5, y_position), 
                   font, font_scale, color, thickness)

def main(image_path, duration):
    # Load image
    image = cv2.imread(image_path)
    
    # Resize image to 1424 x 848
    image = cv2.resize(image, (1424, 848))
    
    # Create a copy of the image for display
    display_image = image.copy()
    
    # Set up OSC client
    client = udp_client.SimpleUDPClient("127.0.0.1", 8000)
    
    while True:
        # Add the info box with initial values
        add_info_box(display_image, 0, 0, 0, duration)
        cv2.imshow("Image with Analysis", display_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('a'):
            # Create a fresh copy of the image for this analysis
            display_image = image.copy()
            
            # Perform SIFT analysis
            contrast, object_count, sizes, proximity, min_size, avg_size, max_size = analyze_image(image, display_image)
            
            # Add information box to the display image
            add_info_box(display_image, contrast, object_count, proximity, duration)
            
            # Send only the object count as an OSC message
            client.send_message("/image/object_count", object_count)
            print(f"Sent OSC: /image/object_count {object_count}")
        
        elif key == ord('b'):
            # Get initial object count before scanning
            _, object_count, _, _, _, _, _ = analyze_image(image, display_image.copy())
            
            # Perform left-to-right scanning effect with a 60x848 rectangle
            scan_width = 60
            scan_height = 848
            num_steps = (image.shape[1] - scan_width) + 1
            step_duration = duration / num_steps
            
            for step in range(num_steps):
                # Calculate the current area to scan
                start_x = step
                end_x = start_x + scan_width
                
                # Invert the colors of the scanned area
                display_image[:, start_x:end_x] = cv2.bitwise_not(display_image[:, start_x:end_x])
                
                # Draw a green cursor at the top of the scanning rectangle
                cv2.rectangle(display_image, (start_x, 0), (end_x, 1), (0, 255, 0), -1)
                
                # Analyze the scanned area for keypoints
                scanned_area = display_image[:, start_x:end_x]
                _, scan_object_count, scan_sizes, _, min_size, avg_size, max_size = analyze_image(scanned_area, scanned_area.copy())
                
                # Transmit object count and size metrics in the scanned area
                client.send_message("/image/scan_object_count", scan_object_count)
                client.send_message("/image/scan_min_size", min_size)
                client.send_message("/image/scan_avg_size", avg_size)
                client.send_message("/image/scan_max_size", max_size)
                print(f"Sent OSC: /image/scan_object_count {scan_object_count}")
                print(f"Sent OSC: /image/scan_min_size {min_size}")
                print(f"Sent OSC: /image/scan_avg_size {avg_size}")
                print(f"Sent OSC: /image/scan_max_size {max_size}")
                
                # Update the info box with scan data and total object count
                add_info_box(display_image, contrast, object_count, proximity, duration, 
                           scan_data=scanned_area.flatten(), scan_object_count=scan_object_count)
                
                # Display the updated image
                cv2.imshow("Image with Analysis", display_image)
                cv2.waitKey(int(step_duration * 1000))
                
                # Remove the green cursor for the next iteration
                display_image[:, start_x:end_x] = cv2.bitwise_not(display_image[:, start_x:end_x])
        
        elif key == 27:  # ESC key to exit
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze graphical score and transmit as OSC data.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument("duration", type=float, help="Total duration of data transmission in seconds.")
    args = parser.parse_args()
    
    main(args.image_path, args.duration)