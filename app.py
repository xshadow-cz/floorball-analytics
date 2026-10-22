import os
import json
import math
import numpy as np
import cv2
import yt_dlp
from ultralytics import YOLO
import streamlit as st
import matplotlib.pyplot as plt

# ==========================================
# 1. VIDEO INGESTION & DOWNLOAD
# ==========================================

def download_or_load_video(video_source: str, save_filename: str = "input_match.mp4") -> str:
    """
    Downloads video from YouTube URL or verifies local file path with cloud bypass headers.
    """
    if os.path.exists(video_source):
        return video_source

    if video_source.startswith("http://") or video_source.startswith("https://"):
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': save_filename,
            'quiet': True,
            'nocheckcertificate': True,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'ios']
                }
            }
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_source])
        return save_filename

    raise FileNotFoundError(f"Video source '{video_source}' not found.")


# ==========================================
# 2. DYNAMIC HOMOGRAPHY & CAMERA MOTION COMPENSATION (CMC)
# ==========================================

class DynamicRinkTransformer:
    def __init__(self, initial_pixel_corners: np.ndarray, court_dimensions: tuple = (40.0, 20.0)):
        """
        initial_pixel_corners: 4 points in pixels for Frame 0 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        court_dimensions: (length_meters, width_meters) - floorball rink is 40m x 20m
        """
        w, h = court_dimensions
        self.target_corners = np.array([
            [0.0, 0.0],   # Top-Left
            [w, 0.0],     # Top-Right
            [w, h],       # Bottom-Right
            [0.0, h]      # Bottom-Left
        ], dtype=np.float32)

        self.initial_corners = initial_pixel_corners.astype(np.float32)
        self.current_pixel_corners = initial_pixel_corners.astype(np.float32)
        self.H, _ = cv2.findHomography(self.current_pixel_corners, self.target_corners)
        
        self.prev_gray = None
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def is_scene_cut(self, current_gray: np.ndarray, threshold: float = 40.0) -> bool:
        """
        Detects hard video cuts, title screens, or transitions using mean absolute pixel difference.
        """
        if self.prev_gray is None:
            return False
        
        diff = cv2.absdiff(self.prev_gray, current_gray)
        mean_diff = np.mean(diff)
        return mean_diff > threshold

    def update_camera_motion(self, current_frame: np.ndarray):
        """
        Tracks static background feature points between consecutive frames 
        and updates current pixel corners and homography matrix H.
        """
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return

        # Handle scene cuts safely
        if self.is_scene_cut(gray):
            self.prev_gray = gray
            return

        mask = np.ones_like(self.prev_gray, dtype=np.uint8) * 255
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=10, mask=mask)

        if p0 is not None and len(p0) > 10:
            p1, st_pts, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)

            if p1 is not None and st_pts is not None:
                good_prev = p0[st_pts == 1]
                good_curr = p1[st_pts == 1]

                M, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr)

                if M is not None and np.sum(inliers) > 8:
                    corners_homogeneous = np.hstack([self.current_pixel_corners, np.ones((4, 1))])
                    new_corners = np.dot(M, corners_homogeneous.T).T
                    self.current_pixel_corners = new_corners.astype(np.float32)

                    new_H, _ = cv2.findHomography(self.current_pixel_corners, self.target_corners)
                    if new_H is not None:
                        self.H = new_H

        self.prev_gray = gray

    def to_meters(self, px: float, py: float) -> tuple:
        """
        Transforms pixel point to pitch meters using updated frame Homography.
        """
        pt = np.array([[[px, py]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        return float(transformed[0][0][0]), float(transformed[0][0][1])


# ==========================================
# 3. TACTICAL ANALYTICS ENGINE
# ==========================================

class FloorballTacticalEngine:
    def __init__(self, goal_a_pos=(0.0, 10.0), goal_b_pos=(40.0, 10.0)):
        self.goal_a = goal_a_pos
        self.goal_b = goal_b_pos
        
        self.possession_counter = {"Team A": 0, "Team B": 0, "Neutral": 0}
        self.passes = {"Team A": 0, "Team B": 0}
        self.shots = []
        
        self.last_possession_player = None
        self.last_possession_team = None

    def calculate_xg(self, shot_x: float, shot_y: float, target_goal: tuple) -> float:
        gx, gy = target_goal
        dist = math.hypot(shot_x - gx, shot_y - gy)
        angle = math.atan2(abs(shot_y - gy), max(dist, 0.1))
        
        raw_xg = 1.0 / (1.0 + math.exp(0.35 * dist + 1.2 * angle - 0.5))
        return round(min(max(raw_xg, 0.01), 0.95), 2)

    def process_frame(self, frame_idx: int, players: list, ball_pos: tuple):
        current_team = "Neutral"
        closest_player = None
        min_dist = float('inf')

        if ball_pos is not None:
            bx, by = ball_pos
            for p in players:
                px, py = p["pos"]
                dist = math.hypot(px - bx, py - by)
                if dist < min_dist and dist <= 1.2:
                    min_dist = dist
                    closest_player = p["id"]
                    current_team = p["team"]

        self.possession_counter[current_team] += 1

        if current_team in ["Team A", "Team B"]:
            if (self.last_possession_team == current_team and 
                self.last_possession_player is not None and 
                self.last_possession_player != closest_player):
                self.passes[current_team] += 1
            
            self.last_possession_team = current_team
            self.last_possession_player = closest_player

        if ball_pos is not None and current_team != "Neutral":
            bx, by = ball_pos
            target_goal = self.goal_b if current_team == "Team A" else self.goal_a
            dist_to_goal = math.hypot(bx - target_goal[0], by - target_goal[1])

            if dist_to_goal <= 12.0 and min_dist <= 0.8:
                if not self.shots or (frame_idx - self.shots[-1]["frame"]) > 30:
                    xg_val = self.calculate_xg(bx, by, target_goal)
                    self.shots.append({
                        "team": current_team,
                        "x": bx,
                        "y": by,
                        "xg": xg_val,
                        "frame": frame_idx
                    })

    def get_summary_report(self) -> dict:
        total_frames = sum(self.possession_counter.values())
        possession_pct = {}
        if total_frames > 0:
            for team, count in self.possession_counter.items():
                possession_pct[team] = round((count / total_frames) * 100, 1)

        team_xg = {"Team A": 0.0, "Team B": 0.0}
        shot_counts = {"Team A": 0, "Team B": 0}

        for shot in self.shots:
            t = shot["team"]
            team_xg[t] = round(team_xg[t] + shot["xg"], 2)
            shot_counts[t] += 1

        return {
            "possession_percentage": possession_pct,
            "completed_passes": self.passes,
            "total_shots": shot_counts,
            "expected_goals_xg": team_xg,
            "shot_details": self.shots
        }


# ==========================================
# 4. MAIN COMPUTER VISION PIPELINE
# ==========================================

def run_match_analysis(video_source: str, model_weights: str = "yolov8n.pt", frame_limit: int = 150) -> dict:
    video_path = download_or_load_video(video_source)
    cap = cv2.VideoCapture(video_path)

    initial_pixel_corners = np.array([
        [150, 200],   # Top-Left corner
        [1130, 200],  # Top-Right corner
        [1250, 680],  # Bottom-Right corner
        [30, 680]     # Bottom-Left corner
    ], dtype=np.float32)

    transformer = DynamicRinkTransformer(initial_pixel_corners)
    model = YOLO(model_weights)
    engine = FloorballTacticalEngine()

    frame_idx = 0

    while cap.isOpened() and frame_idx < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

        transformer.update_camera_motion(frame)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        players = []
        ball_pos = None

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                cls_ids = boxes.cls.int().cpu().tolist()
                xyxy_list = boxes.xyxy.cpu().tolist()

                for tid, cls_id, bbox in zip(track_ids, cls_ids, xyxy_list):
                    cx_px = (bbox[0] + bbox[2]) / 2.0
                    cy_px = bbox[3]

                    x_m, y_m = transformer.to_meters(cx_px, cy_px)

                    if 0.0 <= x_m <= 40.0 and 0.0 <= y_m <= 20.0:
                        if cls_id == 0:  # Player class
                            team = "Team A" if tid % 2 == 0 else "Team B"
                            players.append({"id": tid, "team": team, "pos": (x_m, y_m)})

                        elif cls_id == 3:  # Ball class
                            ball_pos = (x_m, y_m)

        engine.process_frame(frame_idx, players, ball_pos)
        frame_idx += 1

    cap.release()
    return engine.get_summary_report()


# ==========================================
# 5. STREAMLIT FRONTEND DASHBOARD
# ==========================================

def main():
    st.set_page_config(page_title="Floorball Match Analytics", layout="wide")
    st.title("🏑 Automated Floorball Tactical Dashboard")
    st.write("Upload a match video or enter a YouTube URL to extract tactical statistics.")

    # Drag & Drop File Uploader
    uploaded_file = st.file_uploader("Upload a video file from your computer (.mp4):", type=["mp4", "mov", "avi"])
    
    # YouTube URL or Local File Path Text Input
    url_input = st.text_input("OR Enter YouTube URL / File Path:", "")

    video_input = None

    if uploaded_file is not None:
        temp_filename = "temp_uploaded_video.mp4"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.read())
        video_input = temp_filename
    elif url_input.strip() != "":
        video_input = url_input.strip()

    if st.button("🚀 Process Match Video"):
        if not video_input:
            st.warning("Please upload a video file or enter a YouTube URL first!")
        else:
            with st.spinner("Analyzing match frames and tracking camera motion..."):
                try:
                    stats = run_match_analysis(video_input)
                    st.session_state["stats"] = stats
                    st.success("Analysis complete!")
                except Exception as e:
                    st.error(f"Error processing video: {e}")

    # Display results if available
    if "stats" in st.session_state:
        stats = st.session_state["stats"]

        col1, col2, col3, col4 = st.columns(4)
        poss = stats["possession_percentage"]
        col1.metric("Possession", f"Team A {poss.get('Team A', 0)}% - {poss.get('Team B', 0)}% Team B")
        
        passes = stats["completed_passes"]
        col2.metric("Passes Completed", f"Team A: {passes.get('Team A', 0)} | Team B: {passes.get('Team B', 0)}")
        
        shots = stats["total_shots"]
        col3.metric("Shots Taken", f"Team A: {shots.get('Team A', 0)} | Team B: {shots.get('Team B', 0)}")
        
        xg = stats["expected_goals_xg"]
        col4.metric("Expected Goals (xG)", f"Team A: {xg.get('Team A', 0.0)} | Team B: {xg.get('Team B', 0.0)}")

        st.divider()

        st.subheader("🎯 Shot Locations & Calculated xG")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 20)
        ax.set_facecolor('#dcebe6')

        # Draw Rink
        ax.plot([0, 40, 40, 0, 0], [0, 0, 20, 20, 0], color='white', linewidth=3)
        ax.plot([20, 20], [0, 20], color='white', linewidth=2)
        ax.add_patch(plt.Rectangle((0, 8.25), 2.5, 3.5, color='orange', alpha=0.5))
        ax.add_patch(plt.Rectangle((37.5, 8.25), 2.5, 3.5, color='orange', alpha=0.5))

        for shot in stats["shot_details"]:
            color = 'blue' if shot["team"] == "Team A" else 'red'
            size = shot["xg"] * 300 + 50
            ax.scatter(shot["x"], shot["y"], s=size, c=color, alpha=0.7, edgecolors='black')
            ax.annotate(f"xG {shot['xg']}", (shot["x"] + 0.5, shot["y"] + 0.5), fontsize=8)

        ax.set_title("Floorball Court Shot Map (Blue = Team A, Red = Team B)")
        ax.set_xlabel("Length (Meters)")
        ax.set_ylabel("Width (Meters)")
        
        st.pyplot(fig)

        with st.expander("📄 View Full JSON Metrics"):
            st.json(stats)


if __name__ == "__main__":
    main()