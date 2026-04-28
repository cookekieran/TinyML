import serial
import pyautogui
import time

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
THRESHOLD = 5.0  # logit score threshold

def main():
    print(f"Music Hand Gesture Controller")
    hand_in_frame = False  # default = no_hand
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        ser.flushInput()
        ser.write(b'AT+RUNIMPULSE\r\n')
        print("Searching for hand...")

        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if " hand: " in line or line.startswith("hand:"):
                try:
                    score = float(line.split("hand:")[1].strip())
                    
                    if score > THRESHOLD:
                        if not hand_in_frame:
                            print(f"\n Hand Detected! Score: {score}")
                            print("Toggling Music...")
                            pyautogui.press('playpause')
                            hand_in_frame = True 
                    else:
                        if hand_in_frame:
                            print("\n Hand removed.")
                            hand_in_frame = False # reset to status quo (no hand)
                        
                        print(f"Hand Score: {score} (Resting)       ", end='\r')
                        
                except:
                    pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'ser' in locals(): ser.close()

if __name__ == "__main__":
    main()