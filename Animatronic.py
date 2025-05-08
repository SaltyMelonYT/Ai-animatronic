import cv2
import base64
import requests
import time
import json
import random
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llava"

cap = cv2.VideoCapture(0)

# Load memory from file
def load_memory():
    if os.path.exists("memories.json"):
        with open("memories.json", "r") as f:
            return json.load(f)
    return []

# Save memory to file
def save_memory():
    with open("memories.json", "w") as f:
        json.dump(memory, f)

# Initialize memory
memory = load_memory()

# Update analyze_and_adapt to save memory
def analyze_and_adapt(response):
    global memory
    memory.append(response)
    if len(memory) > 100:  # Limit memory size
        memory.pop(0)
    save_memory()
    return response

# Process multiple frames
frame_sequence = []
sequence_length = 5

# Debugging: Notify when capturing frames
print("[DEBUG] Starting frame capture...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[DEBUG] Frame capture failed. Retrying...")
        continue

    print("[DEBUG] Frame captured successfully.")

    # Size for capture, X Y
    frame = cv2.resize(frame, (512, 384))
    print("[DEBUG] Frame resized to 512x384.")

    # Save and encode the frame
    cv2.imwrite("/home/Jonathan/Documents/frame.jpeg", frame)
    print("[DEBUG] Frame saved as frame.jpeg.")

    with open("/home/Jonathan/Documents/frame.jpeg", "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')
        print("[DEBUG] Frame encoded to Base64.")

    # Add frame to sequence
    frame_sequence.append(image_b64)
    if len(frame_sequence) > sequence_length:
        frame_sequence.pop(0)
        print("[DEBUG] Frame sequence updated. Oldest frame removed.")

    # Send it to the AI model
    print("[DEBUG] Sending frame sequence to AI model...")
    payload = {
        "model": MODEL,
        "prompt": "What do you see",
        "images": frame_sequence,
        "stream": True
    }

    response = requests.post(OLLAMA_URL, json=payload)
    print("[DEBUG] Response received from AI model.")

    response_text = ""  # Initialize an empty string to accumulate the response

    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            response_text += data.get("response", "") + " "  # Append each part of the response

    response_text = " ".join(response_text.split()).replace(" ,", ",").replace(" .", ".")  # Normalize spaces and fix punctuation
    print("[DEBUG] AI response processed.")
    print("AI says:", response_text)

    # Prevent the model from being used too much and allow processing time
    print("[DEBUG] Waiting for 30 seconds before next frame capture...")
    time.sleep(30)