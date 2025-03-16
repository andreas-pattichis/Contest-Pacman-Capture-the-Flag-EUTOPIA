# EUTOPIA Pacman Contest <img src="https://github.com/andreaspattichis/Contest-Pacman-Capture-the-Flag-EUTOPIA/assets/63289392/0281e11e-a884-49b9-a97c-784b1d00c622" alt="image" style="width: 8%; vertical-align: middle;"/>

![EUTOPIA Contest Banner](https://github.com/andreaspattichis/Contest-Pacman-Capture-the-Flag-EUTOPIA/assets/63289392/cd32914f-7b98-4268-9600-3048d7f1280b)

## Overview
The EUTOPIA Pacman Contest is a competitive team-based challenge where participants develop intelligent agents for a capture-the-flag variant of Pacman. Based on Berkeley's CS188 course and adapted by UPF with contributions from RMIT and UoM, this Python-based framework provides a platform for implementing and testing advanced AI strategies.

<p align="center">
  <img src="https://github.com/andreaspattichis/Contest-Pacman-Capture-the-Flag-EUTOPIA/assets/63289392/77cf8503-fb4f-4a60-9c90-f77477ac0b5e" alt="image" style="width: 30%;"/>
  <img src="https://github.com/andreaspattichis/Contest-Pacman-Capture-the-Flag-EUTOPIA/assets/63289392/a331b61d-dff2-4f54-9c87-dc6305b5eb46" alt="image" style="width: 30%;"/>
  <img src="https://github.com/andreaspattichis/Contest-Pacman-Capture-the-Flag-EUTOPIA/assets/63289392/5f23367d-e2c0-4aee-bce1-9df7b9d3adbe" alt="image" style="width: 30%;"/>
</p>

## Game Rules
- **Map Structure**: The playing field is divided into two halves, with each team defending their side while invading the opponent's territory.
- **Objective**: Collect food pellets from the opponent's side and return them safely to your territory.
- **Scoring**: Each food pellet returned to your territory earns one point.
- **Victory Conditions**: Win by collecting all opponent's food or having the highest score when time expires.
- **Constraints**: Players must manage computational resources efficiently to ensure timely actions.

## Repository Content
This repository provides implementations of advanced AI agents for the EUTOPIA Pacman Contest:

### Getting Started
1. Clone the EUTOPIA contest repository:
   ```bash
   git clone https://github.com/aig-upf/pacman-eutopia
   ```
2. Replace `team_name_X` with your team's name
3. Add the desired version file (v1.py or v2.py) to your repository

## Agent Implementations

### Version 1 (v1.py)
- **OffensiveQLearningAgent**: Employs Q-learning for strategic offensive play with dynamic action selection, feature-based decision making, and Bayesian inference for tracking opponents.
- **DefensiveReflexCaptureAgent**: Focuses on territory defense using patrol strategies around key chokepoints, with Bayesian belief distributions to track invaders when not directly visible.

### Version 2 (v2.py)
- **OffensiveQLearningAgent**: Enhanced Q-learning agent with weight persistence and adaptive strategy development.
- **DefensiveQLearningAgent**: Q-learning defender that focuses on protecting territory and intercepting invaders, with patrol strategies around key entry points and Bayesian inference for enemy tracking.

## Pre-trained Weights
The repository includes optimized weights for both offensive and defensive agents, enabling immediate high-performance gameplay without training:
- `default_offensive_weights.json`: Optimized weights for offensive strategies
- `defensive_agent_weights_613de35e-a3b5-4f6b-bbe5-579884aa87f0.json`: Refined weights for defensive play

## Key Features
- **Reinforcement Learning**: Both versions leverage Q-learning to improve performance over time.
- **Bayesian Inference**: Used for probabilistic tracking of opponents even when not directly visible.
- **Adaptive Strategies**: Agents dynamically adjust between offense and defense based on game state.
- **Strategic Patrolling**: Defensive agents identify and monitor key map locations to intercept invaders efficiently.
