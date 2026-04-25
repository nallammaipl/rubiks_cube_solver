import cv2
import numpy as np
import time

class SimpleRubikDetector:
    def __init__(self):
        # Open camera
        self.cap = cv2.VideoCapture(0)
        
        # Try different camera indices if 0 doesn't work
        if not self.cap.isOpened():
            print("Camera 0 not found, trying camera 1...")
            self.cap = cv2.VideoCapture(1)
        
        if not self.cap.isOpened():
            raise Exception("Could not open any camera. Please check your camera connection.")
        
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.square_size = 350
        
        # Color ranges (HSV)
        self.colors = {
            'White': ([0, 0, 200], [180, 30, 255], (255, 255, 255)),
            'Yellow': ([20, 80, 180], [35, 255, 255], (0, 255, 255)),
            'Red': ([0, 120, 120], [10, 255, 255], (0, 0, 255)),
            'Red2': ([170, 120, 120], [180, 255, 255], (0, 0, 255)),
            'Green': ([40, 50, 50], [80, 255, 255], (0, 255, 0)),
            'Blue': ([100, 60, 60], [130, 255, 255], (255, 0, 0)),
            'Orange': ([5, 100, 180], [18, 255, 255], (0, 165, 255)),
            'Orange2': ([170, 100, 180], [180, 255, 255], (0, 165, 255))
        }
        
        self.grid_colors = [['?' for _ in range(3)] for _ in range(3)]
        
    def get_color_name(self, hsv_pixel):
        """Determine color name from HSV values"""
        h, s, v = hsv_pixel
        
        # Special handling for white (low saturation, high value)
        if s < 40 and v > 200:
            return 'White', 1.0
        
        # Special handling for orange (high saturation, specific hue)
        if 80 < s and 180 < v:
            if (5 <= h <= 18) or (h >= 170):
                return 'Orange', 1.0
        
        # Check each color range
        best_color = 'Unknown'
        best_confidence = 0
        
        for color_name, (lower, upper, _) in self.colors.items():
            if '2' in color_name:  # Skip secondary ranges, we'll check separately
                continue
            
            # Check primary range
            if (lower[0] <= h <= upper[0] and
                lower[1] <= s <= upper[1] and
                lower[2] <= v <= upper[2]):
                confidence = 1.0
                return color_name, confidence
            
            # Check secondary ranges for red and orange
            if color_name == 'Red':
                if (170 <= h <= 180 and s >= 120 and v >= 120):
                    return 'Red', 1.0
            elif color_name == 'Orange':
                if (170 <= h <= 180 and s >= 100 and v >= 180):
                    return 'Orange', 1.0
        
        # If no exact match, try to find closest based on saturation and value
        if s < 50 and v > 150:
            return 'White', 0.7
        elif s > 80 and v > 150:
            if h < 20 or h > 170:
                return 'Red', 0.6
            elif 20 <= h < 40:
                return 'Yellow', 0.6
            elif 40 <= h < 80:
                return 'Green', 0.6
            elif 80 <= h < 130:
                return 'Blue', 0.6
            elif 130 <= h < 170:
                return 'Orange', 0.6
        
        return 'Unknown', 0.3
    
    def detect_cell(self, roi, frame):
        """Detect color in a single cell"""
        if roi[0] >= roi[2] or roi[1] >= roi[3]:
            return 'Unknown', 0
        
        # Extract ROI
        cell = frame[roi[1]:roi[3], roi[0]:roi[2]]
        
        if cell.size == 0:
            return 'Unknown', 0
        
        # Convert to HSV
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        
        # Get the most common color in the cell (use mode/average)
        hsv_reshaped = hsv.reshape(-1, 3)
        
        # Use median to avoid outliers
        median_hsv = np.median(hsv_reshaped, axis=0)
        
        # Detect color
        color, confidence = self.get_color_name(median_hsv)
        
        return color, confidence
    
    def detect_all(self, frame, cells):
        """Detect all 9 cells"""
        for cell in cells:
            color, conf = self.detect_cell(cell['roi'], frame)
            self.grid_colors[cell['row']][cell['col']] = color
    
    def get_cells(self, frame):
        """Get the 9 cell regions"""
        h, w = frame.shape[:2]
        
        # Calculate square position
        x1 = w//2 - self.square_size//2
        y1 = h//2 - self.square_size//2
        x2 = w//2 + self.square_size//2
        y2 = h//2 + self.square_size//2
        
        # Calculate cell size
        cell_w = self.square_size // 3
        cell_h = self.square_size // 3
        
        cells = []
        for row in range(3):
            for col in range(3):
                # Add padding to avoid grid lines
                pad = 8
                roi_x1 = x1 + col*cell_w + pad
                roi_y1 = y1 + row*cell_h + pad
                roi_x2 = x1 + (col+1)*cell_w - pad
                roi_y2 = y1 + (row+1)*cell_h - pad
                
                cells.append({
                    'row': row,
                    'col': col,
                    'roi': (roi_x1, roi_y1, roi_x2, roi_y2),
                    'center': (x1 + col*cell_w + cell_w//2,
                              y1 + row*cell_h + cell_h//2)
                })
        
        return cells, (x1, y1, x2, y2)
    
    def draw_grid(self, frame, cells, square_coords):
        """Draw the grid and detected colors"""
        x1, y1, x2, y2 = square_coords
        
        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        
        # Draw outer square
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw grid lines
        cell_w = self.square_size // 3
        cell_h = self.square_size // 3
        
        for i in range(1, 3):
            cv2.line(frame, (x1 + i*cell_w, y1), (x1 + i*cell_w, y2), (0, 255, 0), 2)
            cv2.line(frame, (x1, y1 + i*cell_h), (x2, y1 + i*cell_h), (0, 255, 0), 2)
        
        # Draw color indicators
        color_map = {
            'White': (255, 255, 255),
            'Yellow': (0, 255, 255),
            'Red': (0, 0, 255),
            'Green': (0, 255, 0),
            'Blue': (255, 0, 0),
            'Orange': (0, 165, 255),
            'Unknown': (128, 128, 128)
        }
        
        for cell in cells:
            row, col = cell['row'], cell['col']
            color = self.grid_colors[row][col]
            cx, cy = cell['center']
            
            # Get color for display
            bgr = color_map.get(color, (128, 128, 128))
            
            # Draw colored rectangle
            cv2.rectangle(frame, (cx-25, cy-20), (cx+25, cy+20), bgr, -1)
            cv2.rectangle(frame, (cx-25, cy-20), (cx+25, cy+20), (0, 0, 0), 2)
            
            # Display color name
            display_text = color[:3].upper() if color != 'Unknown' else '?'
            cv2.putText(frame, display_text, (cx-15, cy+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return frame
    
    def print_grid(self):
        """Print the current grid to console"""
        print("\n" + "="*50)
        print("Current Rubik's Cube Face:")
        print("-"*50)
        
        for row in self.grid_colors:
            print("|", end="")
            for color in row:
                if color == 'White':
                    print("  ⚪  ", end="")
                elif color == 'Orange':
                    print("  🟠  ", end="")
                elif color == 'Red':
                    print("  🔴  ", end="")
                elif color == 'Green':
                    print("  🟢  ", end="")
                elif color == 'Blue':
                    print("  🔵  ", end="")
                elif color == 'Yellow':
                    print("  🟡  ", end="")
                else:
                    print(f"  {color[:1]}  ", end="")
                print("|", end="")
            print()
            print("-"*50)
    
    def run(self):
        print("\n" + "="*60)
        print("   RUBIK'S CUBE COLOR DETECTOR - SIMPLE VERSION")
        print("="*60)
        print("\n✅ Camera opened successfully!")
        print("\n📌 HOW TO USE:")
        print("   1. Position your Rubik's cube in front of the camera")
        print("   2. Align ONE FACE inside the GREEN SQUARE")
        print("   3. Press 'D' to detect colors")
        print("   4. Press 'A' for auto-detect mode")
        print("\n🎮 CONTROLS:")
        print("   'D' - Detect colors (manual)")
        print("   'A' - Auto-detect mode (continuous)")
        print("   '+/-' - Make square bigger/smaller")
        print("   'S' - Save screenshot")
        print("   'Q' - Quit")
        print("\n" + "="*60)
        
        auto_mode = False
        last_detect = 0
        
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break
            
            # Get cells
            cells, square_coords = self.get_cells(frame)
            
            # Auto detect if enabled
            if auto_mode:
                current_time = time.time()
                if current_time - last_detect > 0.5:
                    self.detect_all(frame, cells)
                    last_detect = current_time
                    self.print_grid()
            
            # Draw everything
            display = self.draw_grid(frame, cells, square_coords)
            
            # Add text overlay
            cv2.putText(display, "RUBIK'S CUBE DETECTOR", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            mode_text = "AUTO MODE [ON]" if auto_mode else "MANUAL MODE"
            mode_color = (0, 255, 0) if auto_mode else (0, 255, 255)
            cv2.putText(display, mode_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)
            
            cv2.putText(display, "Press D=Detect | A=Auto | +/-=Resize | Q=Quit",
                       (10, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Rubik\'s Cube Detector', display)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('d') or key == ord('D'):
                print("\n🔍 Detecting colors...")
                self.detect_all(frame, cells)
                self.print_grid()
                print("✅ Detection complete!")
            elif key == ord('a') or key == ord('A'):
                auto_mode = not auto_mode
                print(f"\n🔄 Auto-detect mode: {'ON' if auto_mode else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                self.square_size = min(500, self.square_size + 20)
                print(f"Square size: {self.square_size}")
            elif key == ord('-'):
                self.square_size = max(200, self.square_size - 20)
                print(f"Square size: {self.square_size}")
            elif key == ord('s') or key == ord('S'):
                filename = f"rubik_{int(time.time())}.png"
                cv2.imwrite(filename, display)
                print(f"📸 Screenshot saved: {filename}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    try:
        detector = SimpleRubikDetector()
        detector.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your camera is connected and not being used by another app")
        print("2. Try closing other apps that might be using the camera (Zoom, Teams, etc.)")
        print("3. Check if you have the correct camera index (try changing 0 to 1 or 2)")