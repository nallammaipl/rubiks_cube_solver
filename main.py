import cv2
import numpy as np
import time
import json
import os
from collections import Counter

# Try to import kociemba
try:
    import kociemba
    KOCIEMBA_AVAILABLE = True
except ImportError:
    KOCIEMBA_AVAILABLE = False
    print("⚠️ Kociemba not available. Run: pip install kociemba")

class RubikCubeSolver:
    def __init__(self):
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            print("Trying camera 1...")
            self.cap = cv2.VideoCapture(1)
            
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        self.square_size = 400
        
        # Color ranges
        self.colors = {
            'White': ([0, 0, 200], [180, 30, 255]),
            'Yellow': ([20, 80, 180], [35, 255, 255]),
            'Red': ([0, 120, 120], [10, 255, 255]),
            'Red2': ([170, 120, 120], [180, 255, 255]),
            'Green': ([40, 50, 50], [80, 255, 255]),
            'Blue': ([100, 60, 60], [130, 255, 255]),
            'Orange': ([5, 100, 180], [18, 255, 255]),
            'Orange2': ([170, 100, 180], [180, 255, 255])
        }
        
        # Store faces
        self.faces = {
            'U': {'name': 'TOP', 'center': 'White', 'colors': [['?' for _ in range(3)] for _ in range(3)], 'captured': False},
            'D': {'name': 'BOTTOM', 'center': 'Red', 'colors': [['?' for _ in range(3)] for _ in range(3)], 'captured': False},
            'F': {'name': 'FRONT', 'center': 'Yellow', 'colors': [['?' for _ in range(3)] for _ in range(3)], 'captured': False},
            'B': {'name': 'BACK', 'center': 'Green', 'colors': [['?' for _ in range(3)] for _ in range(3)], 'captured': False},
            'L': {'name': 'LEFT', 'center': 'Orange', 'colors': [['?' for _ in range(3)] for _ in range(3)], 'captured': False},
            'R': {'name': 'RIGHT', 'center': 'Blue', 'colors': [['?' for _ in range(3)] for _ in range(3)], 'captured': False}
        }
        
        self.current_face = 'U'
        self.face_order = ['U', 'D', 'F', 'B', 'L', 'R']
        self.face_index = 0
        self.current_colors = [['?' for _ in range(3)] for _ in range(3)]
        
        # Solution tracking
        self.solution_moves = []
        self.current_move_index = 0
        self.solving_mode = False
        
        os.makedirs('cube_saves', exist_ok=True)
        
    def get_color_name(self, hsv_pixel):
        h, s, v = hsv_pixel
        
        if s < 40 and v > 200:
            return 'White'
        
        if 80 < s and 180 < v:
            if (5 <= h <= 18) or (h >= 170):
                return 'Orange'
        
        for color_name, (lower, upper) in self.colors.items():
            if '2' in color_name:
                continue
            
            if (lower[0] <= h <= upper[0] and
                lower[1] <= s <= upper[1] and
                lower[2] <= v <= upper[2]):
                return color_name
            
            if color_name == 'Red':
                if (170 <= h <= 180 and s >= 120 and v >= 120):
                    return 'Red'
            elif color_name == 'Orange':
                if (170 <= h <= 180 and s >= 100 and v >= 180):
                    return 'Orange'
        
        return 'Unknown'
    
    def detect_cell(self, roi, frame):
        if roi[0] >= roi[2] or roi[1] >= roi[3]:
            return 'Unknown'
        
        cell = frame[roi[1]:roi[3], roi[0]:roi[2]]
        
        if cell.size == 0:
            return 'Unknown'
        
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        hsv_reshaped = hsv.reshape(-1, 3)
        median_hsv = np.median(hsv_reshaped, axis=0)
        color = self.get_color_name(median_hsv)
        
        return color
    
    def detect_all_cells(self, frame, cells):
        detected_colors = [['?' for _ in range(3)] for _ in range(3)]
        for cell in cells:
            color = self.detect_cell(cell['roi'], frame)
            detected_colors[cell['row']][cell['col']] = color
        return detected_colors
    
    def get_cells(self, frame):
        h, w = frame.shape[:2]
        
        x1 = w//2 - self.square_size//2
        y1 = h//2 - self.square_size//2
        x2 = w//2 + self.square_size//2
        y2 = h//2 + self.square_size//2
        
        cell_w = self.square_size // 3
        cell_h = self.square_size // 3
        
        cells = []
        for row in range(3):
            for col in range(3):
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
    
    def draw_interface(self, frame, cells, square_coords):
        x1, y1, x2, y2 = square_coords
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        cell_w = self.square_size // 3
        cell_h = self.square_size // 3
        
        for i in range(1, 3):
            cv2.line(frame, (x1 + i*cell_w, y1), (x1 + i*cell_w, y2), (0, 255, 0), 2)
            cv2.line(frame, (x1, y1 + i*cell_h), (x2, y1 + i*cell_h), (0, 255, 0), 2)
        
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
            color = self.current_colors[row][col]
            cx, cy = cell['center']
            
            bgr = color_map.get(color, (128, 128, 128))
            cv2.rectangle(frame, (cx-25, cy-20), (cx+25, cy+20), bgr, -1)
            cv2.rectangle(frame, (cx-25, cy-20), (cx+25, cy+20), (0, 0, 0), 2)
            
            display_text = color[:3].upper() if color != 'Unknown' else '?'
            cv2.putText(frame, display_text, (cx-15, cy+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return frame
    
    def draw_move_guide(self, frame, move):
        """Draw visual guide for the current move"""
        if not move:
            return frame
        
        h, w = frame.shape[:2]
        
        # Create overlay for move guide
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-250, 10), (w-10, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Parse move
        face = move[0]
        modifier = move[1] if len(move) > 1 else ''
        
        # Get direction arrow
        if modifier == '2':
            direction = "180°"
            arrow = "⟷"
        elif modifier == "'":
            direction = "Counter-Clockwise"
            arrow = "↺"
        else:
            direction = "Clockwise"
            arrow = "↻"
        
        # Face names
        face_names = {
            'U': 'UP', 'D': 'DOWN', 'F': 'FRONT',
            'B': 'BACK', 'L': 'LEFT', 'R': 'RIGHT'
        }
        
        # Display move information
        cv2.putText(frame, "NEXT MOVE:", (w-240, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display the move big
        move_text = f"{face}{modifier}"
        cv2.putText(frame, move_text, (w-230, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        cv2.putText(frame, f"{face_names.get(face, face)} Face", (w-240, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Turn {direction}", (w-240, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, arrow, (w-240, 195),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw arrow on the cube visualization
        # Show which face to turn with a colored border
        if face == 'U':
            cv2.rectangle(frame, (w//2 - self.square_size//2, 0), 
                         (w//2 + self.square_size//2, self.square_size//3), 
                         (0, 255, 255), 5)
        elif face == 'D':
            cv2.rectangle(frame, (w//2 - self.square_size//2, h - self.square_size//3),
                         (w//2 + self.square_size//2, h), (0, 255, 255), 5)
        elif face == 'F':
            cv2.rectangle(frame, (w//2 - self.square_size//2, h//2 - self.square_size//2),
                         (w//2 + self.square_size//2, h//2 + self.square_size//2), (0, 255, 255), 5)
        
        return frame
    
    def draw_progress(self, frame):
        """Draw solution progress bar"""
        if not self.solution_moves:
            return frame
        
        total = len(self.solution_moves)
        current = self.current_move_index
        
        progress = int((current / total) * 100) if total > 0 else 0
        
        h, w = frame.shape[:2]
        
        # Progress bar background
        bar_width = 300
        bar_height = 20
        bar_x = 10
        bar_y = h - 40
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Progress fill
        fill_width = int((current / total) * bar_width) if total > 0 else 0
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     (0, 255, 0), -1)
        
        # Progress text
        cv2.putText(frame, f"Move {current}/{total} ({progress}%)", 
                   (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Current and remaining moves
        if current < total:
            remaining = total - current
            cv2.putText(frame, f"Remaining: {remaining} moves", 
                       (bar_x + bar_width + 10, bar_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame
    
    def build_kociemba_string(self):
        """Build the cube string for Kociemba solver"""
        color_to_kociemba = {
            'White': 'U', 'Blue': 'R', 'Yellow': 'F',
            'Red': 'D', 'Orange': 'L', 'Green': 'B'
        }
        
        face_order = ['U', 'R', 'F', 'D', 'L', 'B']
        cube_parts = []
        
        for face in face_order:
            if face == 'U':
                your_face = self.faces['U']
            elif face == 'R':
                your_face = self.faces['R']
            elif face == 'F':
                your_face = self.faces['F']
            elif face == 'D':
                your_face = self.faces['D']
            elif face == 'L':
                your_face = self.faces['L']
            elif face == 'B':
                your_face = self.faces['B']
            else:
                cube_parts.extend(['U'] * 9)
                continue
            
            if your_face['captured']:
                colors = your_face['colors']
                for row in range(3):
                    for col in range(3):
                        color = colors[row][col]
                        k_letter = color_to_kociemba.get(color, 'U')
                        cube_parts.append(k_letter)
            else:
                cube_parts.extend(['U'] * 9)
        
        return ''.join(cube_parts)
    
    def get_solution(self):
        """Get optimal solution using Kociemba"""
        if not KOCIEMBA_AVAILABLE:
            return None
        
        try:
            cube_string = self.build_kociemba_string()
            solution = kociemba.solve(cube_string)
            moves = solution.split()
            return moves
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def show_3d_cube(self):
        """Create 3D-like cube visualization with move highlighting"""
        img_size = 1200
        cell_size = 100
        cube_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
        
        color_bgr = {
            'White': (255, 255, 255),
            'Yellow': (0, 255, 255),
            'Red': (0, 0, 255),
            'Green': (0, 255, 0),
            'Blue': (255, 0, 0),
            'Orange': (0, 165, 255),
            'Unknown': (128, 128, 128)
        }
        
        positions = {
            'U': (cell_size * 3, 0, 'TOP (White)'),
            'L': (0, cell_size * 3, 'LEFT (Orange)'),
            'F': (cell_size * 3, cell_size * 3, 'FRONT (Yellow)'),
            'R': (cell_size * 6, cell_size * 3, 'RIGHT (Blue)'),
            'B': (cell_size * 9, cell_size * 3, 'BACK (Green)'),
            'D': (cell_size * 3, cell_size * 6, 'BOTTOM (Red)')
        }
        
        # Determine which face to highlight
        highlight_face = None
        if self.solving_mode and self.current_move_index < len(self.solution_moves):
            move = self.solution_moves[self.current_move_index]
            move_face = move[0]
            # Map move face to our face ID
            face_map = {'U': 'U', 'D': 'D', 'F': 'F', 'B': 'B', 'L': 'L', 'R': 'R'}
            highlight_face = face_map.get(move_face)
        
        for face_id, (start_x, start_y, face_name) in positions.items():
            if not self.faces[face_id]['captured']:
                for row in range(3):
                    for col in range(3):
                        x = start_x + col * cell_size
                        y = start_y + row * cell_size
                        cv2.rectangle(cube_img, (x, y), (x + cell_size, y + cell_size), (200, 200, 200), 2)
                cv2.putText(cube_img, "?", (start_x + cell_size//2 - 10, start_y + cell_size//2 + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 2)
            else:
                colors = self.faces[face_id]['colors']
                
                for row in range(3):
                    for col in range(3):
                        x = start_x + col * cell_size
                        y = start_y + row * cell_size
                        color_name = colors[row][col]
                        
                        bgr = color_bgr.get(color_name, (100, 100, 100))
                        cv2.rectangle(cube_img, (x, y), (x + cell_size, y + cell_size), bgr, -1)
                        cv2.rectangle(cube_img, (x, y), (x + cell_size, y + cell_size), (0, 0, 0), 2)
                        
                        if row == 1 and col == 1:
                            cv2.rectangle(cube_img, (x, y), (x + cell_size, y + cell_size), (0, 255, 255), 3)
                        
                        text = color_name[:3] if color_name != 'Unknown' else '?'
                        text_x = x + cell_size//3
                        text_y = y + cell_size//2 + 5
                        cv2.putText(cube_img, text, (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Highlight the face to turn
            if highlight_face == face_id:
                cv2.rectangle(cube_img, (start_x - 5, start_y - 5), 
                            (start_x + cell_size*3 + 5, start_y + cell_size*3 + 5), 
                            (0, 255, 255), 8)
            
            label_x = start_x + 10
            label_y = start_y - 10
            cv2.putText(cube_img, face_name, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Add solution info if in solving mode
        if self.solving_mode and self.solution_moves:
            cv2.putText(cube_img, f"MOVE: {self.current_move_index + 1}/{len(self.solution_moves)}", 
                       (img_size//2 - 100, img_size - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(cube_img, "YOUR RUBIK'S CUBE", (img_size//2 - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return cube_img
    
    def validate_cube(self):
        """Validate the entire cube"""
        print("\n" + "="*60)
        print("CUBE VALIDATION")
        print("="*60)
        
        all_valid = True
        for face_id, face_data in self.faces.items():
            if face_data['captured']:
                center = face_data['colors'][1][1]
                expected = face_data['center']
                if center != expected:
                    print(f"❌ {face_data['name']}: Center is {center}, should be {expected}")
                    all_valid = False
                else:
                    print(f"✅ {face_data['name']}: {center}")
        
        if not all_valid:
            return False
        
        all_colors = []
        for face_data in self.faces.values():
            if face_data['captured']:
                for row in face_data['colors']:
                    all_colors.extend(row)
        
        color_counts = Counter(all_colors)
        print("\n📊 Color Distribution:")
        for color in ['White', 'Yellow', 'Red', 'Green', 'Blue', 'Orange']:
            count = color_counts.get(color, 0)
            print(f"   {color}: {count}")
        
        return True
    
    def print_instructions(self):
        captured = sum(1 for f in self.faces.values() if f['captured'])
        print("\n" + "="*60)
        print(f"PROGRESS: {captured}/6 FACES")
        if captured < 6:
            current = self.faces[self.current_face]
            print(f"CURRENT: {current['name']} (Center: {current['center']})")
        print("="*60)
        print("CONTROLS:")
        print("  [SPACE] - Capture current face")
        print("  [R]     - Retake current face")
        print("  [V]     - View 3D cube")
        print("  [C]     - Validate cube")
        print("  [S]     - SOLVE (get solving algorithm)")
        print("  [N]     - Next move (during solving mode)")
        print("  [Q]     - Quit")
        print("="*60)
    
    def run(self):
        print("\n" + "🎯"*35)
        print("     RUBIK'S CUBE SOLVER")
        print("     Step-by-Step Visual Guide")
        print("🎯"*35)
        
        if not KOCIEMBA_AVAILABLE:
            print("\n⚠️ KOCIEMBA NOT INSTALLED!")
            print("Run: pip install kociemba")
        
        auto_detect = True
        last_detect = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # If in solving mode, show solution guidance
            if self.solving_mode and self.solution_moves:
                # Display current move guide
                current_move = self.solution_moves[self.current_move_index] if self.current_move_index < len(self.solution_moves) else None
                
                if current_move:
                    frame = self.draw_move_guide(frame, current_move)
                
                # Show 3D cube visualization
                cube_vis = self.show_3d_cube()
                cv2.imshow('Solution Guide - 3D Cube', cube_vis)
                
                # Display solution status
                status_text = f"Step {self.current_move_index + 1}/{len(self.solution_moves)}: {current_move}"
                cv2.putText(frame, status_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Progress bar
                frame = self.draw_progress(frame)
                
                # Instructions for solving mode
                cv2.putText(frame, "Press 'N' for NEXT MOVE | 'Q' to quit solving", 
                           (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                # Normal capture mode
                current_time = time.time()
                if auto_detect and (current_time - last_detect) > 0.3:
                    cells, _ = self.get_cells(frame)
                    self.current_colors = self.detect_all_cells(frame, cells)
                    last_detect = current_time
                
                cells, square_coords = self.get_cells(frame)
                display = self.draw_interface(frame, cells, square_coords)
                frame = display
            
            captured_count = sum(1 for f in self.faces.values() if f['captured'])
            current_face = self.faces[self.current_face]
            
            cv2.putText(frame, f"CAPTURE: {current_face['name']} (Center: {current_face['center']})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Progress: {captured_count}/6", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            if not self.solving_mode:
                detected_center = self.current_colors[1][1]
                center_color = (0, 255, 0) if detected_center == current_face['center'] else (0, 0, 255)
                cv2.putText(frame, f"Center: {detected_center}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, center_color, 2)
                
                solver_status = "Kociemba Ready" if KOCIEMBA_AVAILABLE else "Kociemba Not Installed"
                cv2.putText(frame, solver_status, (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255) if KOCIEMBA_AVAILABLE else (0, 0, 255), 1)
                
                cv2.putText(frame, "SPACE=Capture | R=Retake | S=Solve | V=View | Q=Quit",
                           (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(frame, f"SOLVING MODE - Move {self.current_move_index + 1}/{len(self.solution_moves)}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('Rubik\'s Cube Solver', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                if self.solving_mode:
                    self.solving_mode = False
                    self.solution_moves = []
                    self.current_move_index = 0
                    cv2.destroyWindow('Solution Guide - 3D Cube')
                    print("\n✅ Exited solving mode")
                else:
                    break
                    
            elif key == ord('n') or key == ord('N'):  # Next move in solving mode
                if self.solving_mode and self.solution_moves:
                    if self.current_move_index < len(self.solution_moves) - 1:
                        self.current_move_index += 1
                        print(f"\n👉 Next move: {self.solution_moves[self.current_move_index]}")
                        print(f"   Progress: {self.current_move_index + 1}/{len(self.solution_moves)}")
                    else:
                        print("\n🎉 CONGRATULATIONS! CUBE SHOULD BE SOLVED! 🎉")
                        print("   Press Q to exit solving mode")
                        self.solving_mode = False
                        self.solution_moves = []
                        self.current_move_index = 0
                        cv2.destroyWindow('Solution Guide - 3D Cube')
            
            elif not self.solving_mode:
                if key == ord(' '):
                    if captured_count < 6:
                        detected_center = self.current_colors[1][1]
                        expected_center = current_face['center']
                        
                        if detected_center == expected_center:
                            print(f"\n✅ Captured {current_face['name']}")
                            self.faces[self.current_face]['colors'] = [row[:] for row in self.current_colors]
                            self.faces[self.current_face]['captured'] = True
                            
                            if self.face_index < len(self.face_order) - 1:
                                self.face_index += 1
                                self.current_face = self.face_order[self.face_index]
                                self.print_instructions()
                            else:
                                print("\n" + "🎉"*35)
                                print("ALL FACES CAPTURED! Press 'S' for solution")
                                print("🎉"*35)
                        else:
                            print(f"\n❌ Wrong face! Expected {expected_center}, got {detected_center}")
                    else:
                        print("\n✅ All faces captured! Press 'S' for solution")
                        
                elif key == ord('r') or key == ord('R'):
                    if captured_count < 6:
                        print(f"\n🔄 Retaking {current_face['name']}")
                        self.faces[self.current_face]['colors'] = [['?' for _ in range(3)] for _ in range(3)]
                        self.faces[self.current_face]['captured'] = False
                        self.print_instructions()
                    else:
                        print("\n🔄 Resetting all faces...")
                        for face_id in self.faces:
                            self.faces[face_id]['colors'] = [['?' for _ in range(3)] for _ in range(3)]
                            self.faces[face_id]['captured'] = False
                        self.face_index = 0
                        self.current_face = 'U'
                        self.print_instructions()
                        
                elif key == ord('v') or key == ord('V'):
                    cube_3d = self.show_3d_cube()
                    cv2.imshow('Your 3D Rubik\'s Cube', cube_3d)
                    cv2.waitKey(0)
                    cv2.destroyWindow('Your 3D Rubik\'s Cube')
                
                elif key == ord('c') or key == ord('C'):
                    self.validate_cube()
                
                elif key == ord('s') or key == ord('S'):  # SOLVE
                    if captured_count == 6:
                        if self.validate_cube():
                            print("\n🔧 Generating OPTIMAL solution...")
                            moves = self.get_solution()
                            if moves:
                                self.solution_moves = moves
                                self.current_move_index = 0
                                self.solving_mode = True
                                print(f"\n✅ Solution generated! {len(moves)} moves")
                                print(f"👉 First move: {moves[0]}")
                                print("\n📖 HOW TO USE:")
                                print("   1. Look at the highlighted face on the 3D cube")
                                print("   2. Follow the arrow direction shown on screen")
                                print("   3. Press 'N' after completing each move")
                                print("   4. Keep going until all moves are done!\n")
                                
                                # Show 3D cube window
                                cube_vis = self.show_3d_cube()
                                cv2.imshow('Solution Guide - 3D Cube', cube_vis)
                            else:
                                print("\n❌ Could not generate solution")
                        else:
                            print("\n⚠️ Invalid cube state. Press R to recapture")
                    else:
                        print(f"\n⚠️ Need {6-captured_count} more faces")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    try:
        app = RubikCubeSolver()
        app.print_instructions()
        app.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()