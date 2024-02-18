# EUTOPIA Pacman Contest <img src="https://github.com/andreaspattichis/Contest-Pacman-Capture-the-Flag-EUTOPIA/assets/63289392/0281e11e-a884-49b9-a97c-784b1d00c622" alt="image" style="width: 10%; vertical-align: middle;"/>




<img src="https://github.com/andreaspattichis/Contest-Pacman-Capture-the-Flag-EUTOPIA/assets/63289392/cd32914f-7b98-4268-9600-3048d7f1280b" alt="image" style="width: 75%; float: left; margin-right: 10px;"/>


---
## Introduction

The EUTOPIA Pacman Contest is an exciting team-based competition where participants from different universities create agents to play a multiplayer capture-the-flag version of Pacman. It's a collaborative project based on Berkeley's CS188 Intro to AI course and has been adapted for this contest by UPF, with contributions from RMIT and UoM. The entire codebase is in Python and is modular, catering to different levels of participation.

---
## Contest Framework

There are three main components to the framework:

- **Agent Development**: Teams will create their own repositories for their agents using the provided basic template for behavior.
- **Local Tournament**: Scripts for running custom local tournaments are includexd.
- **UPF Tournament**: The official module used by UPF to organize tournaments. Participants do not need to contribute to this module.

---
## Rules Overview
<p align="center">
  <img src="https://github.com/andreaspattichis/Contest-Pacman-Capture-the-Flag-EUTOPIA/assets/63289392/9d97ad19-1cb3-4094-82f7-d689d7ceead8" alt="image" style="width: 30%;"/>
  <img src="https://github.com/andreaspattichis/Contest-Pacman-Capture-the-Flag-EUTOPIA/assets/63289392/1e318a12-d304-4d8a-8dc3-e387b09d3f81" alt="image" style="width: 30%;"/>
  <img src="https://github.com/andreaspattichis/Contest-Pacman-Capture-the-Flag-EUTOPIA/assets/63289392/d7f8193a-bbed-44f7-827c-dccdd6b698f7" alt="image" style="width: 30%;"/>
</p>


- **Layout**: The map is divided into two halves, with agents needing to defend and capture food on their respective sides.
- **Scoring**: Points are earned by returning food pellets to your side.
- **Gameplay**: Eating Pacman, power capsules, and observations come with specific rules to enhance strategic gameplay.
- **Winning**: The game can end by collecting almost all opponent's dots or by having the highest score when the move limit is reached.
- **Computation Time**: Timely decisions are crucial, with strict rules for computation time to ensure smooth tournament flow.

---
## Repository Contents

The `/versions/` folder contains Python code for teams' implementations of defensive and offensive agents. To use these, you need to:

1. Clone the GitHub project for the EUTOPIA contest:
   'git clone https://github.com/aig-upf/pacman-eutopia'
2. Replace `team_name_X` with your team's name.
3. Add the python file of the version you want to test and experiment with and then follow the instructions of the EUTOPIA's GitHub Repo.

---
## Versions In Detail:
### Version 1 (v1.py) Detailed Description
#### A. OffensiveQLearningAgent
- **Strategy**: Leverages Q-learning for dynamic action selection to optimize offensive gameplay.
- **Trained Weights Loading**: Capable of loading pre-trained weights, enhancing decision-making from past experiences.
- **Q-learning Algorithm Parameters**: Adjusts exploration and exploitation trade-off dynamically, with exploration rate (epsilon) set to 0.15 during training to encourage exploration, and 0 during gameplay for exploitation. The learning rate (alpha) is set at 0.2.
- **Action Selection**: Dynamically selects actions based on the game state and current knowledge, prioritizing returning home when carrying significant food or nearing victory conditions.
- **Feature Extraction**: Extracts relevant features for decision-making, including proximity to ghosts, distance to nearby food, and the urgency of returning home with food.
- **Weighted Q-value Computation**: Calculates Q-values based on a weighted combination of features and learned weights, forming the decision-making basis.
- **Weight Updating**: Updates knowledge by adjusting weights using the Q-learning update rule, considering observed rewards, discount factors, and predicted future rewards.
- **Reward Calculation**: Rewards are calculated based on various factors, including proximity to enemies, progress towards objectives, score changes, and distance to the nearest food.
- **Persistence and Learning**: Retains and builds upon learned knowledge across multiple game sessions by persisting trained weights.
- **Ghost Position Inference**: Utilizes Bayesian inference to estimate ghost positions based on observations and time, enhancing offensive tactics.

#### B. DefensiveReflexCaptureAgent
- **Patrol Behavior**: Focuses on strategic map points, particularly chokepoints, for effective area defense.
- **Belief Distribution**: Uses Bayesian inference for estimating potential invader locations, informing patrol adjustments.
- **Chokepoint Identification**: Analyzes map layout to identify and position near key areas, boosting defensive capabilities.
- **Feature-Based Decision Making**: Decisions are informed by features like invader presence and distances, among others.
- **Weighted Features**: Prioritizes certain defensive actions through feature weighting, like maintaining defense and targeting invaders.
- **Defensive Strategies**: Designs to be a formidable defender, intercepting enemy Pac-Man and securing the territory.

## Version 2 (v2.py) Detailed Description
### A. OffensiveQLearningAgent
- **Strategy**: Utilizes Q-learning to dynamically select actions for optimizing offensive gameplay.
- **Trained Weights Loading**: Capable of loading weights from a file, utilizing past learning experiences to enhance decision-making.
- **Q-learning Algorithm Parameters**: Adjusts exploration-exploitation trade-off dynamically. Sets exploration rate (epsilon) to 0.15 during training for exploration, and 0 during gameplay for exploitation. Learning rate (alpha) is set at 0.2.
- **Action Selection**: Prioritizes returning home when carrying significant food amounts or nearing victory conditions.
- **Feature Extraction**: Extracts relevant features for decision-making, including ghost proximity, nearby food distance, and urgency to return home with food.
- **Weighted Q-value Computation**: Computes Q-values based on a weighted combination of features and learned weights, guiding decision-making.
- **Weight Updating**: Adjusts weights using the Q-learning update rule, factoring in observed rewards, discount factors, and predicted future rewards.
- **Reward Calculation**: Factors in proximity to enemies, score changes, and progress towards objectives for reward calculation.
- **Persistence and Learning**: Retains and builds upon learned knowledge over multiple sessions by persisting trained weights.
- **Ghost Position Inference**: Employs Bayesian inference to estimate ghost positions based on observations and time, enhancing offensive strategy.

### DefensiveQLearningAgent
- **Strategy**: Employs Q-learning for optimizing defensive gameplay, focusing on protecting territory and intercepting invaders.
- **Epsilon-Greedy Action Selection**: Uses an epsilon-greedy strategy for action selection, balancing exploration and exploitation.
- **Weight Initialization and Loading**: Initializes or loads weights for feature importance, facilitating adaptive strategy development.
- **Feature-Based Decision Making**: Extracts and utilizes features such as invader distance, food defending, and entry point proximity for decision-making.
- **Reward System**: Configures rewards based on state transitions, considering factors like food protected, invaders intercepted, and personal safety.
- **Learning and Adaptation**: Updates weights based on the learning rate and observed rewards, allowing for strategy refinement over time.
- **Bayesian Inference for Enemy Positioning**: Utilizes Bayesian inference to update beliefs about enemy locations, improving defensive positioning.
- **Patrol and Intercept Strategy**: Implements a patrol strategy around key entry points, adjusting dynamically based on inferred enemy movements.
- **Performance Tracking and Weight Persistence**: Tracks cumulative rewards and persists learning weights to file for continued improvement across games.


## Getting Started

Download the source code, install dependencies, and refer to the `Getting Started` section to run a game. Use provided options to customize the game execution, record games, and review logs and replays.

