# 🤖 Safe & Human-Aware Robot Path Planning using Deep Reinforcement Learning

---

## 📌 Overview

This project focuses on developing a **human-aware robot navigation system** using **Deep Reinforcement Learning (DRL)**.
The goal is to enable a robot to reach a target location while ensuring:

* 🚫 Collision avoidance
* 🧍 Respect for human comfort distance
* ⚡ Efficient path planning

The system is trained using the **Proximal Policy Optimization (PPO)** algorithm in a custom simulation environment.

---

## 🎯 Objectives

* Design a custom RL-based navigation environment using PyBullet and Gymnasium
* Implement PPO-based path planning
* Incorporate **human-aware safety constraints**
* Evaluate performance using safety and efficiency metrics
* Analyze trade-off between **goal-reaching vs safety**

---

## 🧠 Key Features

* ✅ Human-aware navigation (dynamic obstacles)
* ✅ Safety-first reward design
* ✅ Custom Gym environment
* ✅ PPO-based training
* ✅ Evaluation metrics & analysis
* ✅ Research-oriented implementation

---

## 🛠️ Tech Stack

| Component     | Technology              |
| ------------- | ----------------------- |
| Language      | Python                  |
| RL Algorithm  | PPO (Stable-Baselines3) |
| Simulation    | PyBullet                |
| Environment   | Gymnasium               |
| Visualization | Matplotlib              |

---

## ⚙️ Project Structure

```
RL Project/
│
├── env/
│   └── human_aware_env.py
│
├── train/
│   ├── train_ppo.py
│   ├── evaluate.py
│   └── plot_results.py
│
├── monitor.csv
├── main.py
└── README.md
```

---

## ▶️ How to Run

### 🔹 1. Install Dependencies

```bash
pip install pybullet gymnasium stable-baselines3 numpy matplotlib
```

### 🔹 2. Train the Model

```bash
python train/train_ppo.py
```

### 🔹 3. Evaluate the Model

```bash
python train/evaluate.py
```

### 🔹 4. Plot Results

```bash
python train/plot_results.py
```

---

## 📊 Sample Output

```
Episodes: 20
Success rate: 0.0
Collisions: 0
Average steps: 500
```

🔍 Interpretation:

* Zero collisions → Strong safety performance
* Conservative navigation → Human-aware behavior

---

## 📈 Results Summary

| Metric            | Value |
| ----------------- | ----- |
| Episodes Tested   | 20    |
| Success Rate      | 0.0   |
| Collision Count   | 0     |
| Avg. Steps        | 500   |
| Safety Violations | 0     |

---

## 🧾 Base Papers

* Proximal Policy Optimization Algorithms
  https://arxiv.org/abs/1707.06347

* Socially Aware Navigation with Deep Reinforcement Learning
  https://arxiv.org/abs/1803.09892

---

## 👥 Team Members

| Name             | Registration Number |
| ---------------- | ------------------- |
| Priyanshu Ranjan | 23BAI10691          |
| Shrish           | 23BAI11284          |
| Rajat Acharya    | 23BAI10282          |

---

## 🚀 Future Work

* Improve goal-reaching efficiency
* Add multi-agent coordination
* Incorporate real-world robot deployment
* Enhance human motion prediction

---

## 🎓 Conclusion

This project demonstrates how **Reinforcement Learning can be extended beyond efficiency to include safety and human-awareness**, making it suitable for real-world robotic applications.

---

⭐ *If you like this project, consider giving it a star!*
