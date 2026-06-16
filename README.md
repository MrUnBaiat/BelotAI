# Multi-Agent PPO for Belot (Trick-Taking Card Game)

A high-performance, vectorized implementation of **Recurrent MAPPO (Multi-Agent Proximal Policy Optimization)** designed to master the complex, imperfect-information trick-taking card game, Belot. 

This repository leverages an in-process lockstep vectorization architecture, a Centralized Critic with Perfect Global Information (CTDE), and a custom belief-state heuristic matrix to train highly strategic game-playing agents.

---

## Key Achievements
* **Robust Codebase:** Features a comprehensive `pytest` test suite boasting **92% code coverage** ensuring rigorous logic validation across complex card mechanics.
* **Strong Emergent Play:** Successfully trained cooperative/competitive behavior. Agents have achieved excellent gameplay performance against reference baselines.
* **Next Milestone:** Currently developing an interactive inference engine to evaluate agent performance in live matches against real human players (Belot.md).

---

## Architecture & Features

### 1. Vectorization Framework (`train.py`, `vec_env.py`)
* **In-Process Lockstep Vectorization:** Avoids multi-processing overhead by managing $N$ independent `BelotEnv` instances inside a single process. It batches the expensive network forward passes by exposing the active agent's observation across all environments simultaneously.
* **Precise Hidden-State Routing:** Manages sequential dependencies over asymmetric turns by tracking, splitting, and routing LSTM hidden states `(h, c)` on a per-seat, per-environment basis.
* **Clean Episode Boundaries:** Discards incomplete, in-flight games at the end of a collection budget. This ensures that stored trajectories represent complete games where the GAE bootstrap value is unconditionally $0.0$, eliminating truncation bias.

### 2. Algorithmic Implementation (`model.py`, `memory.py`)
* **Recurrent MAPPO:** Combines shared-parameter Actor-Critic networks with an LSTM layer in the Actor to capture historical context over the course of an 8-trick hand. Especially useful for card counting.
* **Centralized Training, Decentralized Execution (CTDE):** 
  * **The Actor (`model.py`)** evaluates **Imperfect Local Information** (513-dimensional vector containing private cards, known board state, and tracking metrics) to output legal action distributions.
  * **The Critic (`model.py`)** evaluates **Perfect Global Information** (332-dimensional vector containing absolute card locations and opponent hands) for low-variance state-value estimation during training.
* **Dense Reward with Final True-Up:** Implements zero-sum step rewards scaled by trick values ($162$ points max). At the end of the episode, a terminal retroaction loop backs up true match-point goals ($16$ points max) into the trajectory, correcting rewards so the episode sum exactly matches the zero-sum strategic target.

---

## Training Performance & Diagnostics

### Key Metric
* **Eval/PointDiffVsRandom:** Measures the true match-point differential per game. The trained policy achieves an average margin of **+5.5 points** out of a 16-point game maximum, demonstrating a decisive statistical dominance over the baseline.
![alt text](image.png)
