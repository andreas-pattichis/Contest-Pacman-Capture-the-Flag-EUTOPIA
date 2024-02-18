# EUTOPIA Pacman Contest

## Introduction

The EUTOPIA Pacman Contest is an exciting team-based competition where participants from different universities create agents to play a multiplayer capture-the-flag version of Pacman. It's a collaborative project based on Berkeley's CS188 Intro to AI course and has been adapted for this contest by UPF, with contributions from RMIT and UoM. The entire codebase is in Python and is modular, catering to different levels of participation.

## Contest Framework

There are three main components to the framework:

- **Agent Development**: Teams will create their own repositories for their agents using the provided basic template for behavior.
- **Local Tournament**: Scripts for running custom local tournaments are includexd.
- **UPF Tournament**: The official module used by UPF to organize tournaments. Participants do not need to contribute to this module.

## Rules Overview

- **Layout**: The map is divided into two halves, with agents needing to defend and capture food on their respective sides.
- **Scoring**: Points are earned by returning food pellets to your side.
- **Gameplay**: Eating Pacman, power capsules, and observations come with specific rules to enhance strategic gameplay.
- **Winning**: The game can end by collecting almost all opponent's dots or by having the highest score when the move limit is reached.
- **Computation Time**: Timely decisions are crucial, with strict rules for computation time to ensure smooth tournament flow.

## Repository Contents

The `/versions/` folder contains Python code for teams' implementations of defensive and offensive agents. To use these, you need to:

1. Clone the GitHub project for the EUTOPIA contest:
   'git clone https://github.com/aig-upf/pacman-eutopia'
2. Replace `team_name_X` with your team's name.
3. Add the python file of the version you want to test and experiment with and then follow the instructions of the EUTOPIA's GitHub Repo.

## Getting Started

Download the source code, install dependencies, and refer to the `Getting Started` section to run a game. Use provided options to customize the game execution, record games, and review logs and replays.

