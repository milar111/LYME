import cv2

def find_camera_indices():
    available_indices = []
    for index in range(11):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Try to get the name (works on some systems)
                available_indices.append(index)
                print(f"✅ Camera found at index {index}")
                
                # Show a quick preview to confirm it's the right one
                cv2.imshow(f"Camera Index {index}", frame)
                print(f"   Press 'q' on the window to check next index...")
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
            cap.release()
    
    if not available_indices:
        print("❌ No cameras detected. Check your USB connection.")
    else:
        print(f"\nFinal list of working indices: {available_indices}")

if __name__ == "__main__":
    find_camera_indices()