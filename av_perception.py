
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from sklearn.cluster import DBSCAN
from scipy.optimize import curve_fit
import time
import pandas as pd
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# PART 1: SIMULATION ENVIRONMENT
# ============================================================================

class Vehicle:
    """Ego vehicle with bicycle dynamics"""
    def __init__(self, x=0, y=0, yaw=0, velocity=15.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = velocity
        self.length = 4.5
        self.width = 1.8
        self.wheelbase = 2.8
        
    def update(self, steering_angle, acceleration, dt=0.1):
        steering_angle = np.clip(steering_angle, -0.6, 0.6)
        acceleration = np.clip(acceleration, -5.0, 3.0)
        self.v += acceleration * dt
        self.v = np.clip(self.v, 0, 30)
        if abs(self.v) > 0.1:
            yaw_rate = (self.v / self.wheelbase) * np.tan(steering_angle)
        else:
            yaw_rate = 0
        self.yaw += yaw_rate * dt
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        
    def get_corners(self):
        half_l = self.length / 2
        half_w = self.width / 2
        corners = np.array([[-half_l, -half_w], [half_l, -half_w], 
                           [half_l, half_w], [-half_l, half_w]])
        rot_mat = np.array([[np.cos(self.yaw), -np.sin(self.yaw)],
                           [np.sin(self.yaw), np.cos(self.yaw)]])
        corners = corners @ rot_mat.T
        corners[:,0] += self.x
        corners[:,1] += self.y
        return corners


class Road:
    def __init__(self, lane_width=3.5, num_lanes=2):
        self.lane_width = lane_width
        self.num_lanes = num_lanes
        self.center_coeffs = [0, 0.03, -0.001, 0.00005]
        
    def get_lane_center_y(self, x):
        return np.polyval(self.center_coeffs[::-1], x)
    
    def get_lane_boundaries(self, x):
        center_y = self.get_lane_center_y(x)
        left_y = center_y + self.lane_width / 2
        right_y = center_y - self.lane_width / 2
        return left_y, center_y, right_y


class Obstacle:
    def __init__(self, x, y, vx=0, vy=0, obj_type="car", size=0.5):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.type = obj_type
        self.size = size
        
    def update(self, dt=0.1):
        self.x += self.vx * dt
        self.y += self.vy * dt
        
    def get_distance(self, vehicle_x, vehicle_y):
        return np.sqrt((self.x - vehicle_x)**2 + (self.y - vehicle_y)**2)


class SensorSuite:
    def __init__(self, camera_noise_std=0.05, lidar_noise_std=0.1, 
                 dropout_rate=0.02, max_range=50.0):
        self.camera_noise_std = camera_noise_std
        self.lidar_noise_std = lidar_noise_std
        self.dropout_rate = dropout_rate
        self.max_range = max_range
        
    def get_lane_detections(self, road, vehicle, num_samples=30):
        detections = []
        x_start = vehicle.x
        x_end = vehicle.x + self.max_range
        x_samples = np.linspace(x_start, x_end, num_samples)
        for x in x_samples:
            if np.random.random() < self.dropout_rate:
                continue
            left, _, right = road.get_lane_boundaries(x)
            left_noisy = left + np.random.normal(0, self.camera_noise_std)
            right_noisy = right + np.random.normal(0, self.camera_noise_std)
            detections.append([x, left_noisy])
            detections.append([x, right_noisy])
        return np.array(detections) if detections else np.empty((0, 2))
    
    def get_lidar_points(self, obstacles, vehicle, num_points_per_obstacle=8):
        points = []
        for obs in obstacles:
            dist = obs.get_distance(vehicle.x, vehicle.y)
            if dist > self.max_range:
                continue
            num_pts = np.random.poisson(num_points_per_obstacle)
            for _ in range(num_pts):
                if np.random.random() < self.dropout_rate:
                    continue
                px = obs.x + np.random.normal(0, self.lidar_noise_std)
                py = obs.y + np.random.normal(0, self.lidar_noise_std)
                points.append([px, py])
        num_noise = np.random.poisson(3)
        for _ in range(num_noise):
            angle = np.random.uniform(-np.pi/3, np.pi/3)
            dist = np.random.uniform(5, self.max_range)
            px = vehicle.x + dist * np.cos(angle + vehicle.yaw)
            py = vehicle.y + dist * np.sin(angle + vehicle.yaw)
            points.append([px, py])
        return np.array(points) if points else np.empty((0, 2))


# ============================================================================
# PART 2: PERCEPTION ALGORITHMS
# ============================================================================

class LaneDetector:
    def __init__(self, poly_degree=2, min_points=10):
        self.poly_degree = poly_degree
        self.min_points = min_points
        self.left_coeffs = None
        self.right_coeffs = None
        self.confidence = 0.0
        
    def detect(self, lane_points):
        if len(lane_points) < self.min_points:
            self.left_coeffs = None
            self.right_coeffs = None
            self.confidence = 0.0
            return None, None, 0.0
        median_y = np.median(lane_points[:, 1])
        left_mask = lane_points[:, 1] > median_y
        right_mask = lane_points[:, 1] < median_y
        left_points = lane_points[left_mask]
        right_points = lane_points[right_mask]
        self.left_coeffs = self._fit_poly(left_points)
        self.right_coeffs = self._fit_poly(right_points)
        if self.left_coeffs is not None and self.right_coeffs is not None:
            self.confidence = min(1.0, (len(left_points) + len(right_points)) / 50)
        else:
            self.confidence = 0.3
        return self.left_coeffs, self.right_coeffs, self.confidence
    
    def _fit_poly(self, points):
        if len(points) < 3:
            return None
        x = points[:, 0]
        y = points[:, 1]
        try:
            coeffs = np.polyfit(x, y, self.poly_degree)
            return coeffs
        except:
            return None
    
    def get_lateral_offset(self, vehicle_x, vehicle_y):
        if self.left_coeffs is None or self.right_coeffs is None:
            return None
        left_y = np.polyval(self.left_coeffs, vehicle_x)
        right_y = np.polyval(self.right_coeffs, vehicle_x)
        lane_center = (left_y + right_y) / 2
        return vehicle_y - lane_center


class ObstacleDetector:
    def __init__(self, eps=1.0, min_samples=3):
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.tracks = {}
        self.next_id = 0
        
    def detect(self, lidar_points):
        if len(lidar_points) < self.min_samples:
            return []
        labels = self.dbscan.fit_predict(lidar_points)
        obstacles = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = lidar_points[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            obstacles.append({'position': centroid, 'num_points': len(cluster_points), 'label': label})
        return obstacles
    
    def update_tracking(self, detections, dt=0.1):
        for track_id, track in self.tracks.items():
            track['position'] = track['position'] + track['velocity'] * dt
            track['age'] += dt
            track['last_seen'] += dt
        matched_tracks = set()
        for det in detections:
            det_pos = det['position']
            best_track = None
            best_dist = 2.5
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                dist = np.linalg.norm(det_pos - track['position'])
                if dist < best_dist:
                    best_dist = dist
                    best_track = track_id
            if best_track is not None:
                old_pos = self.tracks[best_track]['position']
                new_vel = (det_pos - old_pos) / dt
                self.tracks[best_track]['velocity'] = 0.7 * self.tracks[best_track]['velocity'] + 0.3 * new_vel
                self.tracks[best_track]['position'] = det_pos
                self.tracks[best_track]['last_seen'] = 0
                matched_tracks.add(best_track)
            else:
                self.tracks[self.next_id] = {
                    'position': det_pos,
                    'velocity': np.array([0.0, 0.0]),
                    'age': 0.0,
                    'last_seen': 0.0,
                    'num_points': det['num_points']
                }
                self.next_id += 1
        self.tracks = {k: v for k, v in self.tracks.items() if v['last_seen'] < 2.0}
        return self.tracks


class LaneKeepingController:
    def __init__(self, kp=0.8, ki=0.05, kd=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        
    def compute_steering(self, lateral_offset, dt=0.1):
        if lateral_offset is None:
            return 0.0
        self.integral += lateral_offset * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)
        derivative = (lateral_offset - self.prev_error) / dt if dt > 0 else 0
        steering = self.kp * lateral_offset + self.ki * self.integral + self.kd * derivative
        steering = np.clip(steering, -0.6, 0.6)
        self.prev_error = lateral_offset
        return steering


# ============================================================================
# PART 3: METRICS COLLECTOR
# ============================================================================

class MetricsCollector:
    def __init__(self):
        self.lateral_errors = []
        self.detection_rates = []
        self.false_positives = []
        self.tracking_errors = []
        self.lane_confidence = []
        self.computation_times = []
        
    def record_frame(self, lateral_error, detection_rate, false_positives, 
                     tracking_error, lane_confidence, comp_time):
        if lateral_error is not None:
            self.lateral_errors.append(abs(lateral_error))
        self.detection_rates.append(detection_rate)
        self.false_positives.append(false_positives)
        if tracking_error is not None:
            self.tracking_errors.append(tracking_error)
        self.lane_confidence.append(lane_confidence)
        self.computation_times.append(comp_time)
        
    def get_statistics(self):
        stats = {}
        if self.lateral_errors:
            stats['lateral_error_mean'] = np.mean(self.lateral_errors)
            stats['lateral_error_std'] = np.std(self.lateral_errors)
            stats['lateral_error_max'] = np.max(self.lateral_errors)
            stats['lateral_error_95th'] = np.percentile(self.lateral_errors, 95)
        else:
            stats['lateral_error_mean'] = float('nan')
        stats['detection_rate_mean'] = np.mean(self.detection_rates)
        stats['detection_rate_std'] = np.std(self.detection_rates)
        stats['false_positives_mean'] = np.mean(self.false_positives)
        stats['false_positives_std'] = np.std(self.false_positives)
        if self.tracking_errors:
            stats['tracking_error_mean'] = np.mean(self.tracking_errors)
            stats['tracking_error_std'] = np.std(self.tracking_errors)
        else:
            stats['tracking_error_mean'] = float('nan')
        stats['comp_time_mean'] = np.mean(self.computation_times) * 1000
        stats['comp_time_std'] = np.std(self.computation_times) * 1000
        stats['fps'] = 1.0 / np.mean(self.computation_times) if self.computation_times else 0
        return stats


# ============================================================================
# PART 4: SIMULATION ENGINE
# ============================================================================

class PerceptionSimulation:
    def __init__(self, config):
        self.config = config
        self.duration = config.get('duration', 30.0)
        self.dt = config.get('dt', 0.1)
        self.vehicle = Vehicle()
        self.road = Road()
        self.sensors = SensorSuite(
            camera_noise_std=config.get('sensor_noise', 0.05),
            lidar_noise_std=config.get('sensor_noise', 0.1),
            dropout_rate=config.get('dropout_rate', 0.02)
        )
        self.lane_detector = LaneDetector()
        self.obstacle_detector = ObstacleDetector(eps=config.get('clustering_eps', 1.0), min_samples=3)
        self.controller = LaneKeepingController(kp=config.get('controller_kp', 0.8))
        self.metrics = MetricsCollector()
        self.obstacles = self._create_obstacles()
        
    def _create_obstacles(self):
        return [
            Obstacle(30.0, -1.2, vx=-3.0, vy=0, obj_type="car", size=0.8),
            Obstacle(55.0, 1.8, vx=1.5, vy=0, obj_type="pedestrian", size=0.4),
            Obstacle(80.0, -0.5, vx=0, vy=0, obj_type="cone", size=0.3),
            Obstacle(120.0, 0.5, vx=-2.0, vy=0.5, obj_type="car", size=0.8),
        ]
        
    def run(self, visualize=False):
        num_steps = int(self.duration / self.dt)
        gt_obstacles = self.obstacles.copy()
        for step in range(num_steps):
            frame_start = time.time()
            lane_points = self.sensors.get_lane_detections(self.road, self.vehicle)
            lidar_points = self.sensors.get_lidar_points(self.obstacles, self.vehicle)
            self.lane_detector.detect(lane_points)
            lateral_offset = self.lane_detector.get_lateral_offset(self.vehicle.x, self.vehicle.y)
            detections = self.obstacle_detector.detect(lidar_points)
            tracks = self.obstacle_detector.update_tracking(detections, self.dt)
            detected_gt = 0
            for gt in gt_obstacles:
                gt_pos = np.array([gt.x, gt.y])
                for track in tracks.values():
                    if np.linalg.norm(track['position'] - gt_pos) < 2.0:
                        detected_gt += 1
                        break
            detection_rate = detected_gt / max(len(gt_obstacles), 1)
            false_pos = 0
            for track in tracks.values():
                min_dist = min([np.linalg.norm(track['position'] - np.array([gt.x, gt.y])) 
                               for gt in gt_obstacles] or [100])
                if min_dist > 2.0:
                    false_pos += 1
            tracking_errors = []
            for gt in gt_obstacles:
                gt_pos = np.array([gt.x, gt.y])
                min_dist = 100
                for track in tracks.values():
                    dist = np.linalg.norm(track['position'] - gt_pos)
                    if dist < min_dist:
                        min_dist = dist
                if min_dist < 2.0:
                    tracking_errors.append(min_dist)
            tracking_error = np.mean(tracking_errors) if tracking_errors else None
            steering = self.controller.compute_steering(lateral_offset, self.dt)
            self.vehicle.update(steering, acceleration=0.0, dt=self.dt)
            for obs in self.obstacles:
                obs.update(self.dt)
            comp_time = time.time() - frame_start
            self.metrics.record_frame(lateral_offset, detection_rate, false_pos,
                                     tracking_error, self.lane_detector.confidence, comp_time)
        return self.metrics.get_statistics()


# ============================================================================
# PART 5: EXPERIMENT RUNNER
# ============================================================================

def run_experiments():
    experiments = [
        {'name': 'Baseline (Low Noise)', 'duration': 30.0, 'dt': 0.1, 
         'sensor_noise': 0.03, 'dropout_rate': 0.01, 'clustering_eps': 0.8, 
         'controller_kp': 0.8, 'num_runs': 10},
        {'name': 'High Sensor Noise', 'duration': 30.0, 'dt': 0.1, 
         'sensor_noise': 0.15, 'dropout_rate': 0.01, 'clustering_eps': 0.8, 
         'controller_kp': 0.8, 'num_runs': 10},
        {'name': 'High Dropout Rate', 'duration': 30.0, 'dt': 0.1, 
         'sensor_noise': 0.05, 'dropout_rate': 0.12, 'clustering_eps': 0.8, 
         'controller_kp': 0.8, 'num_runs': 10},
        {'name': 'Aggressive Clustering', 'duration': 30.0, 'dt': 0.1, 
         'sensor_noise': 0.05, 'dropout_rate': 0.02, 'clustering_eps': 2.0, 
         'controller_kp': 0.8, 'num_runs': 10},
        {'name': 'Conservative Clustering', 'duration': 30.0, 'dt': 0.1, 
         'sensor_noise': 0.05, 'dropout_rate': 0.02, 'clustering_eps': 0.5, 
         'controller_kp': 0.8, 'num_runs': 10},
        {'name': 'Aggressive Control', 'duration': 30.0, 'dt': 0.1, 
         'sensor_noise': 0.05, 'dropout_rate': 0.02, 'clustering_eps': 0.8, 
         'controller_kp': 1.2, 'num_runs': 10},
    ]
    
    all_results = []
    print("=" * 80)
    print("AUTONOMOUS VEHICLE PERCEPTION SYSTEM - EXPERIMENT RESULTS")
    print("=" * 80)
    
    for exp in experiments:
        print(f"\nRunning: {exp['name']}")
        run_results = []
        for run_id in range(exp['num_runs']):
            np.random.seed(run_id * 100)
            sim = PerceptionSimulation(exp)
            stats = sim.run(visualize=False)
            stats['run_id'] = run_id
            stats['config_name'] = exp['name']
            run_results.append(stats)
            print(f"    Run {run_id+1}/{exp['num_runs']}: "
                  f"Lateral err={stats['lateral_error_mean']:.3f}m, "
                  f"Det rate={stats['detection_rate_mean']:.3f}")
        
        all_results.extend(run_results)
        lat_errors = [r['lateral_error_mean'] for r in run_results if not np.isnan(r['lateral_error_mean'])]
        det_rates = [r['detection_rate_mean'] for r in run_results]
        fp_rates = [r['false_positives_mean'] for r in run_results]
        print(f"  -> SUMMARY: Lateral Err: {np.mean(lat_errors):.3f}±{np.std(lat_errors):.3f}m | "
              f"Detection: {np.mean(det_rates):.3f}±{np.std(det_rates):.3f} | "
              f"False Pos: {np.mean(fp_rates):.3f}±{np.std(fp_rates):.3f}")
    
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS BY CONFIGURATION")
    print("=" * 80)
    summary = df.groupby('config_name').agg({
        'lateral_error_mean': ['mean', 'std', 'min', 'max'],
        'detection_rate_mean': ['mean', 'std'],
        'false_positives_mean': ['mean', 'std'],
        'fps': ['mean', 'std']
    }).round(4)
    print(summary)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'perception_results_{timestamp}.csv', index=False)
    summary.to_csv(f'perception_summary_{timestamp}.csv')
    print(f"\nResults saved to: perception_results_{timestamp}.csv")
    return df, summary


# ============================================================================
# PART 6: VISUALIZATION
# ============================================================================

def run_visualization():
    print("\n" + "-" * 80)
    print("Starting interactive visualization...")
    print("Close the plot window to exit")
    print("-" * 80)
    
    config = {'duration': 40.0, 'dt': 0.1, 'sensor_noise': 0.05, 
              'dropout_rate': 0.02, 'clustering_eps': 0.8, 'controller_kp': 0.8}
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 8))
    sim = PerceptionSimulation(config)
    num_steps = int(config['duration'] / config['dt'])
    
    for step in range(num_steps):
        ax.clear()
        lane_points = sim.sensors.get_lane_detections(sim.road, sim.vehicle)
        lidar_points = sim.sensors.get_lidar_points(sim.obstacles, sim.vehicle)
        sim.lane_detector.detect(lane_points)
        lateral_offset = sim.lane_detector.get_lateral_offset(sim.vehicle.x, sim.vehicle.y)
        detections = sim.obstacle_detector.detect(lidar_points)
        tracks = sim.obstacle_detector.update_tracking(detections, sim.dt)
        
        x_range = np.arange(sim.vehicle.x - 10, sim.vehicle.x + 60, 1)
        lefts, centers, rights = [], [], []
        for x in x_range:
            l, c, r = sim.road.get_lane_boundaries(x)
            lefts.append(l)
            centers.append(c)
            rights.append(r)
        ax.plot(x_range, lefts, 'w--', linewidth=2, alpha=0.8)
        ax.plot(x_range, centers, 'y-', linewidth=1.5, alpha=0.6)
        ax.plot(x_range, rights, 'w--', linewidth=2, alpha=0.8)
        
        for obs in sim.obstacles:
            circle = Circle((obs.x, obs.y), obs.size, color='red', alpha=0.7)
            ax.add_patch(circle)
        
        for track in tracks.values():
            circle = Circle((track['position'][0], track['position'][1]), 0.35, 
                           color='lime', fill=False, linewidth=2, alpha=0.8)
            ax.add_patch(circle)
        
        if len(lidar_points) > 0:
            ax.scatter(lidar_points[:,0], lidar_points[:,1], c='green', s=8, alpha=0.4)
        if len(lane_points) > 0:
            ax.scatter(lane_points[:,0], lane_points[:,1], c='blue', s=12, alpha=0.6)
        
        corners = sim.vehicle.get_corners()
        ax.fill(corners[:,0], corners[:,1], 'cyan', alpha=0.9)
        
        info_text = f"Time: {sim.vehicle.x:.1f}m | Speed: {sim.vehicle.v*3.6:.1f} km/h\n"
        info_text += f"Lateral Offset: {lateral_offset:.2f}m" if lateral_offset is not None else "Lateral Offset: N/A"
        info_text += f" | Tracks: {len(tracks)}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlim(sim.vehicle.x - 5, sim.vehicle.x + 55)
        ax.set_ylim(-6, 6)
        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Y (m)", fontsize=12)
        ax.set_title("Autonomous Vehicle Perception - Lane & Obstacle Detection", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#1a1a2e')
        
        steering = sim.controller.compute_steering(lateral_offset, sim.dt)
        sim.vehicle.update(steering, acceleration=0.0, dt=sim.dt)
        for obs in sim.obstacles:
            obs.update(sim.dt)
        
        plt.pause(0.05)
        if not plt.fignum_exists(fig.number):
            break
    
    plt.ioff()
    plt.show()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("VEHICLE ENGINEERING - PERCEPTION SYSTEM FOR AUTONOMOUS VEHICLES")
    print("=" * 80)
    print("Pure Python Implementation | GitHub Ready")
    print("=" * 80)
    
    # Run experiments (no visualization)
    results_df, summary_df = run_experiments()
    
    # Ask user if they want visualization
    print("\n" + "=" * 80)
    response = input("Run interactive visualization? (y/n): ")
    if response.lower() == 'y':
        run_visualization()
    
    print("\n" + "=" * 80)
    print("PROJECT COMPLETE")
    print("=" * 80)
    print("Files generated:")
    print("  - perception_results_*.csv (raw data)")
    print("  - perception_summary_*.csv (statistics)")
    print("\nTo re-run: python av_perception.py")


if __name__ == "__main__":
    main()

# End of code - COPY UNTIL HERE
