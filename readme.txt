# NBA Game Outcome Prediction (Reinforcement Learning)

This project trains a reinforcement learning model using PPO to predict NBA game results. The model outputs:

- Predicted point spread
- Predicted total points
- Predicted win/loss outcome

The environment is built with Gymnasium and uses historical NBA stats. The model is trained with Stable-Baselines3.

---

## How to Run

### 1. Install Dependencies

Run:

    pip install -r requirements.txt

Make sure your `nbaData/` folder contains:

- enhanced_games.csv
- TeamStatistics.csv

---

### 2. Train the Model

Run:

    python train/train_agent.py

This will train the PPO model and save it to `models/nba_rl_predictor.zip`.

---

### 3. Evaluate the Model

Run:

    python evaluate_model.py

This will load the trained model and print predictions for spread, total points, and win/loss along with accuracy stats.
