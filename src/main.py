import numpy as np
import pandas as pd
import gluonts
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.trainer import Trainer
import yfinance as yf

# Download historical stock data for Apple
ticker = "AAPL"
start_date = "2010-01-01"
end_date = "2023-02-25"
data = yf.download(ticker, start=start_date, end=end_date)

# Preprocess the data
data = data["Close"].asfreq("B").fillna(method="ffill")
train_data = ListDataset(
    [{"start": data.index[0], "target": data.values}],
    freq = "B"
)

# Define the reinforcement learning problem
state_space = [data.values, np.arange(10, 100, 10), np.arange(1, 5)]
action_space = [np.arange(10, 100, 10), np.arange(1, 5)]
reward_function = lambda y_true, y_pred: -((y_true - y_pred) ** 2).mean()

# Implement the Q-Learning algorithm
class QLearning:
    def __init__(self, state_space, action_space, learning_rate, discount_factor):
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = np.zeros(shape=(len(state_space),) + tuple(map(len, action_space)))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def get_action(self, state, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            action = np.array([np.random.choice(actions) for actions in self.action_space])
        else:
            state_idx = tuple(np.searchsorted(self.state_space[i], state[i]) for i in range(len(self.state_space)))
            action_idx = np.unravel_index(np.argmax(self.q_table[state_idx]), self.q_table.shape[len(self.state_space)])
            action = np.array([self.action_space[i][idx] for i, idx in enumerate(action_idx)])
        return action

    def update(self, state, action, next_state, reward):
        state_idx = tuple(np.searchsorted(self.state_space[i], state[i]) for i in range(len(self.state_space)))
        action_idx = tuple(np.searchsorted(self.action_space[i], action[i]) for i in range(len(self.action_space)))
        next_state_idx = tuple(np.searchsorted(self.state_space[i], next_state[i]) for i in range(len(self.state_space)))
        self.q_table[state_idx + action_idx] += self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[next_state_idx]) - self.q_table[state_idx + action_idx])

# Train the reinforcement learning agent
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
num_episodes = 1000
agent = QLearning(state_space, action_space, learning_rate, discount_factor)

for i in range(num_episodes):
    obs = train_data.list_data[0]["target"]
    done = False
    while not done:
        state = [obs[-10:], 10, 1]
        action = agent.get_action(state, epsilon)
        estimator = DeepAREstimator(freq="B", prediction_length=5, trainer=Trainer(epochs=5), context_length=action[0], num_layers=action[1])
        predictor = estimator.train(train_data=train_data)
        forecast_it, ts_it = make_evaluation_predictions(train_data, predictor=predictor, num_samples=100)
        forecasts = list(forecast_it)
        tss = list(ts_it)
        y_true = tss[0]["target"][-5:]
        y_pred = forecasts[0].mean[-5:]
        reward = reward_function(y_true, y_pred)
        next_state = [np.concatenate([obs, y_pred]), action[0], action[1]]
        obs = np.concatenate([obs, y_pred[-1:]])
        if len(obs) > 200:
            done = True
        agent.update(state, action, next_state, reward)

# Generate forecasts using the optimal DeepAR model
state = [data.values[-10:], 10, 1]
action = agent.get_action(state, 0)
estimator = DeepAREstimator(freq="B", prediction_length=5, trainer=Trainer(epochs=5), context_length=action[0], num_layers=action[1])
predictor = estimator.train(train_data=train_data)
forecast_it, ts_it = make_evaluation_predictions(train_data, predictor=predictor, num_samples=100)
forecasts = list(forecast_it)
tss = list(ts_it)

# Evaluate the performance
def evaluate_forecasts(tss, forecasts, metric):
    errors = []
    for i in range(len(forecasts)):
        forecast = forecasts[i]
        ts = tss[i]["target"]
        if metric == "mse":
            errors.append(np.mean((forecast.mean - ts) ** 2))
        elif metric == "rmse":
            errors.append(np.sqrt(np.mean((forecast.mean - ts) ** 2)))
        else:
            raise ValueError(f"Unknown metric {metric}")
    return np.mean(errors)

mse = evaluate_forecasts(tss, forecasts, metric="mse")
rmse = evaluate_forecasts(tss, forecasts, metric="rmse")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")


