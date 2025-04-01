import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Step 1: Load Data and Filter for Current Season
# -----------------------------
file_path = r'/Users/anthonyklemm/Downloads/archive (2)/DEV _ March Madness.csv'
df = pd.read_csv(file_path)

# Print the columns to double-check column names
print("Columns in dataset:", df.columns.tolist())

# For this example, we use a subset of columns (adjust these if needed)
cols = ["Season", "Short Conference Name", "Adjusted Temo",
        "Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency",
        "eFGPct", "TOPct", "Mapped Conference Name", "Mapped ESPN Team Name",
        "Full Team Name", "Seed", "Region", "Post-Season Tournament"]
df = df[cols]

# Ensure Season is numeric
df["Season"] = pd.to_numeric(df["Season"], errors='coerce')

# Filter data for the current season (2025 represents the 2024-2025 season)
current_season = 2025
df = df[df["Season"] == current_season]

# Fill missing numeric values (if any)
df.fillna(df.median(numeric_only=True), inplace=True)

# -----------------------------
# Step 2: Derive Additional Features
# -----------------------------
# Calculate an efficiency ratio
df["Efficiency_Ratio"] = df["Adjusted Offensive Efficiency"] / df["Adjusted Defensive Efficiency"]

# Define major conferences (adjust as needed)
big_conferences = ["ACC", "SEC", "B12", "B10", "BE"]

# Compute median efficiency ratio (from current season data)
median_eff_ratio = df["Efficiency_Ratio"].median()

# Derive Upset Propensity: teams not in major conferences with above-median efficiency ratio
def compute_upset_propensity(row):
    if row["Short Conference Name"] not in big_conferences and row["Efficiency_Ratio"] > median_eff_ratio:
        return 1
    else:
        return 0

df["Upset_Propensity"] = df.apply(compute_upset_propensity, axis=1)

# Define target: 1 if the team made "March Madness", else 0.
df["Tournament_Target"] = (df["Post-Season Tournament"].str.strip() == "March Madness").astype(int)

# -----------------------------
# Step 3: Feature Selection and Scaling
# -----------------------------
feature_cols = ["Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency",
                "eFGPct", "TOPct", "Adjusted Temo", "Efficiency_Ratio", "Upset_Propensity"]

scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# -----------------------------
# Step 4: Create Sequences per Team
# -----------------------------
# Because we have season-level aggregated data (one record per team for the current season),
# we set sequence_length to 1.
sequence_length = 1

X, y = [], []
teams = df["Mapped ESPN Team Name"].unique()
for team in teams:
    team_data = df[df["Mapped ESPN Team Name"] == team].sort_values("Season")
    # With season-level data for the current season, each team should have exactly one record.
    if len(team_data) < sequence_length:
        continue
    team_features = team_data[feature_cols].values
    team_target = team_data["Tournament_Target"].values[0]
    # For sequence_length=1, the "sequence" is just that single record.
    X.append(team_features[-sequence_length:])
    y.append(team_target)

X = np.array(X)
y = np.array(y)
print(f"Created {len(X)} sequences of shape {X.shape[1:]} for training.")

# -----------------------------
# Step 5: Build and Train the LSTM Model
# -----------------------------
# Note: With sequence_length=1, the LSTM essentially behaves like a feed-forward network.
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, len(feature_cols))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Since the dataset may be small, we use a higher number of epochs for testing
history = model.fit(X, y, epochs=20, batch_size=4, validation_split=0.2)

# -----------------------------
# Step 6: Evaluate the Model
# -----------------------------
y_pred_prob = model.predict(X).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

accuracy = accuracy_score(y, y_pred)
roc_auc = roc_auc_score(y, y_pred_prob)
logloss = log_loss(y, y_pred_prob)

print("\nCurrent Season LSTM Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Log loss: {logloss:.4f}")

# -----------------------------
# Step 7: (Optional) Head-to-Head Prediction Function
# -----------------------------
def predict_matchup(team_A_name, team_B_name, model, data, sequence_length, feature_cols):
    """
    Predicts the winner of a matchup between two teams based on current season data.
    """
    team_A_data = data[data["Mapped ESPN Team Name"] == team_A_name].sort_values("Season")
    team_B_data = data[data["Mapped ESPN Team Name"] == team_B_name].sort_values("Season")

    if len(team_A_data) < sequence_length or len(team_B_data) < sequence_length:
        raise ValueError("Not enough data to build a sequence for one of the teams.")

    team_A_seq = team_A_data[feature_cols].iloc[-sequence_length:].values.reshape(1, sequence_length, len(feature_cols))
    team_B_seq = team_B_data[feature_cols].iloc[-sequence_length:].values.reshape(1, sequence_length, len(feature_cols))

    prob_A = model.predict(team_A_seq)[0][0]
    prob_B = model.predict(team_B_seq)[0][0]

    # Compare the probabilities to decide a winner:
    win_probability_A = prob_A / (prob_A + prob_B)
    print(f"{team_A_name} win probability: {win_probability_A:.4f}")
    print(f"{team_B_name} win probability: {1 - win_probability_A:.4f}")

    predicted_winner = team_A_name if win_probability_A >= 0.5 else team_B_name
    return predicted_winner
def plot_lstm_probability_heatmap(matchups, lstm_pred_probs, actual_results):
    data = np.array([lstm_pred_probs, actual_results])

    plt.figure(figsize=(12, 6))
    sns.heatmap(data, annot=True, cmap='coolwarm', fmt=".3f", xticklabels=matchups, yticklabels=['Predicted', 'Actual'], cbar=True)
    plt.xlabel('Matchups')
    plt.title('LSTM Predicted vs Actual Probabilities')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# Define interesting matchups manually (just examples, use actual matchups from your bracket)
matchups = [
    "Florida vs Norfolk State",
    "Florida vs Duke",
    "Creighton vs Louisville",
    "Memphis vs Colorado State"
]

# Replace these with actual indices from your dataset matching these teams
matchup_indices = [0, 1, 2, 3]  # Adjust this based on your actual data indices

# Extract predicted probabilities and actual results from your existing arrays
lstm_pred_probs = model.predict(X[matchup_indices]).flatten()
actual_results = y[matchup_indices]

# Call the heatmap visualization function
plot_lstm_probability_heatmap(matchups, lstm_pred_probs, actual_results)

def plot_matchup_predictions(matchups, team_A_probs, team_B_probs):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(matchups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_A = ax.bar(x - width/2, team_A_probs, width, label='Team A', color='steelblue')
    bars_B = ax.bar(x + width/2, team_B_probs, width, label='Team B', color='saddlebrown')

    # Add plain percentage labels (no annotations for winners)
    for bar in bars_A:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9)

    for bar in bars_B:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9)

    ax.set_ylabel('Predicted Win Probability')
    ax.set_title('Model Predictions for Selected Matchups')
    ax.set_xticks(x)
    ax.set_xticklabels(matchups, rotation=30, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.show()



matchups = ["Florida vs Duke", "Florida vs Norfolk St", "Creighton vs Louisville"]
team_A_probs = [0.4998, 0.9460, 0.4377]
team_B_probs = [0.5002, 0.0540, 0.5623]
actual_winners = ["Team B", "Team A", "Team A"]  # Optional: Only if you want green bars for correct picks

plot_matchup_predictions(matchups, team_A_probs, team_B_probs)

from sklearn.metrics import roc_curve, auc

def plot_lstm_roc_curve(y_true, y_pred_prob):
    # Compute false positive rate, true positive rate and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='dodgerblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guessing
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - LSTM Model')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# After evaluating your model, call the function:
plot_lstm_roc_curve(y, y_pred_prob)

# -----------------------------
# Step 8: Interactive Input Loop for Head-to-Head Predictions
# -----------------------------
print("\nModel is ready for head-to-head predictions using current season data!")
print("Type 'quit' at any time to exit.")

while True:
    team_A_input = input("\nEnter the name of Team A (as in 'Mapped ESPN Team Name'): ").strip()
    if team_A_input.lower() == "quit":
        break
    team_B_input = input("Enter the name of Team B: ").strip()
    if team_B_input.lower() == "quit":
        break
    try:
        winner = predict_matchup(team_A_input, team_B_input, model, df, sequence_length, feature_cols)
        print("Predicted Winner:", winner)
    except ValueError as e:
        print("Error:", e)

#%%
