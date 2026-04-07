import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np

GOAL_REWARD = 100.0
COLLISION_PENALTY = -200.0
COMFORT_PENALTY = -20.0
TIME_PENALTY = -1.0
SMOOTHNESS_PENALTY = -0.1

COMFORT_DISTANCE = 1.5
COLLISION_DISTANCE = 0.3

class HumanAwareNavEnv(gym.Env):
    def __init__(self):
        super(HumanAwareNavEnv, self).__init__()

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32
        )

        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.robot_id = None
        self.goal = None
        self.humans = None
        self.max_steps = 500
        self.current_step = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Plane
        p.loadURDF("plane.urdf")

        # Robot
        self.robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.1])

        # Goal
        self.goal = np.array([4.0, 4.0], dtype=np.float32)

        # Humans (dynamic obstacles)
        self.humans = [
            np.array([2.0, 2.0], dtype=np.float32),
            np.array([-2.0, -2.0], dtype=np.float32)
        ]

        # Initial observation (valid but simple)
        obs = np.array([
            0.0, 0.0, 0.0, 0.0,
            np.linalg.norm(self.goal),
            0.0,
            10.0,
            0.0,
            0.0
        ], dtype=np.float32)

        self.current_step = 0

        info = {}
        return obs, info

    def _get_robot_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        vel, _ = p.getBaseVelocity(self.robot_id)

        robot_x, robot_y = pos[0], pos[1]
        linear_velocity = np.linalg.norm([vel[0], vel[1]])

        _, _, yaw = p.getEulerFromQuaternion(orn)

        return robot_x, robot_y, linear_velocity, yaw

    def _distance_and_angle(self, src_x, src_y, tgt_x, tgt_y, heading):
        dx = tgt_x - src_x
        dy = tgt_y - src_y

        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) - heading

        return distance, angle

    def _move_humans(self):
        for i, human in enumerate(self.humans):
            direction = np.random.uniform(0, 2 * np.pi)
            step_size = 0.02

            human[0] += step_size * np.cos(direction)
            human[1] += step_size * np.sin(direction)

            self.humans[i] = human

    def step(self, action):
        self.current_step += 1

        linear_vel = float(action[0])
        angular_vel = float(action[1])

        # Apply robot motion
        p.resetBaseVelocity(
            self.robot_id,
            linearVelocity=[linear_vel, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, angular_vel]
        )

        # Move humans FIRST
        self._move_humans()

        p.stepSimulation()

        # Robot state
        robot_x, robot_y, robot_vel, robot_heading = self._get_robot_state()

        # Goal metrics
        goal_dist, goal_angle = self._distance_and_angle(
            robot_x, robot_y,
            self.goal[0], self.goal[1],
            robot_heading
        )

        # Human metrics
        human_distances = []
        human_angles = []

        for human in self.humans:
            dist, ang = self._distance_and_angle(
                robot_x, robot_y,
                human[0], human[1],
                robot_heading
            )
            human_distances.append(dist)
            human_angles.append(ang)

        nearest_idx = int(np.argmin(human_distances))

        # Final observation
        obs = np.array([
            robot_x,
            robot_y,
            robot_vel,
            robot_heading,
            goal_dist,
            goal_angle,
            human_distances[nearest_idx],
            human_angles[nearest_idx],
            0.02  # human velocity (constant)
        ], dtype=np.float32)

        reward = TIME_PENALTY

        # Distance to nearest human
        nearest_human_dist = human_distances[nearest_idx]

        # Collision check
        if nearest_human_dist < COLLISION_DISTANCE:
            reward += COLLISION_PENALTY
            terminated = True

        # Comfort distance violation
        elif nearest_human_dist < COMFORT_DISTANCE:
            reward += COMFORT_PENALTY

        # Goal reached
        if goal_dist < 0.5:
            reward += GOAL_REWARD
            terminated = True

        # Smoothness penalty (discourage sharp turns)
        reward += SMOOTHNESS_PENALTY * abs(angular_vel)

        terminated = False
        truncated = False
        info = {}

        # Max steps reached
        if self.current_step >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info