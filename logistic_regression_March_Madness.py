import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import re

# -----------------------------
# Step 1: Load Data and Select Columns
# -----------------------------
file_path = r'/Users/anthonyklemm/Downloads/archive (2)/DEV _ March Madness.csv'
df = pd.read_csv(file_path)

print("Columns in dataset:", df.columns.tolist())

cols = ["Season", "Short Conference Name", "Adjusted Temo",
        "Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency",
        "eFGPct", "TOPct", "Mapped Conference Name", "Mapped ESPN Team Name",
        "Full Team Name", "Seed", "Region", "Post-Season Tournament"]

df = df[cols]

# Convert Season to numeric and fill missing numeric values
df["Season"] = pd.to_numeric(df["Season"], errors='coerce')
df.fillna(df.median(numeric_only=True), inplace=True)

# -----------------------------
# Step 2: Filter for 2025 Season
# -----------------------------
current_season = 2025
df = df[df["Season"] == current_season]

# -----------------------------
# Step 3: Derive Additional Features
# -----------------------------
# Efficiency ratio: offensive efficiency divided by defensive efficiency.
df["Efficiency_Ratio"] = df["Adjusted Offensive Efficiency"] / df["Adjusted Defensive Efficiency"]

# Define major conferences (adjust as needed)
big_conferences = ["ACC", "SEC", "B12"]
median_eff_ratio = df["Efficiency_Ratio"].median()

def compute_upset_propensity(row):
    if row["Short Conference Name"] not in big_conferences and row["Efficiency_Ratio"] > median_eff_ratio:
        return 1
    else:
        return 0

df["Upset_Propensity"] = df.apply(compute_upset_propensity, axis=1)

# Define target: 1 if team made "March Madness", else 0.
df["Tournament_Target"] = (df["Post-Season Tournament"].str.strip() == "March Madness").astype(int)

# -----------------------------
# Step 4: Feature Selection and Scaling
# -----------------------------
feature_cols = ["Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency",
                "eFGPct", "TOPct", "Adjusted Temo", "Efficiency_Ratio", "Upset_Propensity"]

scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# -----------------------------
# Step 5: Prepare Data for Logistic Regression
# -----------------------------
# Since each team appears only once for the current season, we use the data directly.
X = df[feature_cols].values
y = df["Tournament_Target"].values

# (Optional) Split data for evaluation purposes.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

# Evaluate the LR model on the test set
lr_pred_prob = log_reg.predict_proba(X_test)[:, 1]
lr_pred = (lr_pred_prob >= 0.5).astype(int)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_roc_auc = roc_auc_score(y_test, lr_pred_prob)
lr_logloss = log_loss(y_test, lr_pred_prob)

print("\nLogistic Regression Evaluation:")
print(f"Accuracy: {lr_accuracy:.4f}")
print(f"ROC-AUC: {lr_roc_auc:.4f}")
print(f"Log loss: {lr_logloss:.4f}")

# -----------------------------
# Step 6: Visualization of Logistic Regression Performance
# -----------------------------
def plot_logistic_sigmoid_curve(model, X_test, y_test):
    # Get decision function (log-odds) from the model
    log_odds = model.decision_function(X_test)
    # Apply sigmoid function to log-odds
    sigmoid_probs = 1 / (1 + np.exp(-log_odds))
    # Sort values for a smooth curve
    sorted_indices = log_odds.argsort()
    sorted_log_odds = log_odds[sorted_indices]
    sorted_sigmoid_probs = sigmoid_probs[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_log_odds, sorted_sigmoid_probs, label='Sigmoid Curve', color='skyblue')
    plt.scatter(log_odds, y_test, color='red', alpha=0.6, label='Actual Results')
    plt.xlabel('Log-Odds (Decision Function)')
    plt.ylabel('Probability')
    plt.title('Logistic Regression Model Sigmoid Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_logistic_sigmoid_curve(log_reg, X_test, y_test)

from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='dodgerblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

plot_roc_curve(y_test, lr_pred_prob)

# -----------------------------
# Step 7: Visualize Efficiency Ratios for Selected Teams
# -----------------------------
teams_of_interest = ["Duke", "BYU", "Florida", "Michigan State", "Utah State", "Florida State"]
# Filter the data for the teams of interest using the "Mapped ESPN Team Name" column
efficiency_teams_df = df[df["Mapped ESPN Team Name"].isin(teams_of_interest)]

plt.figure(figsize=(10,6))
sns.barplot(x="Mapped ESPN Team Name", y="Efficiency_Ratio", data=efficiency_teams_df, palette="viridis")
plt.title("Efficiency Ratios for Selected Teams")
plt.xlabel("Team")
plt.ylabel("Efficiency Ratio")
plt.ylim(0, efficiency_teams_df["Efficiency_Ratio"].max() * 1.1)  # add a bit of headroom
plt.show()

# -----------------------------
# Step 8: Interactive Input Loop for Predictions
# -----------------------------
print("\nModel is ready for head-to-head predictions using the 2025 season data!")
print("Type 'quit' at any time to exit.")

def predict_matchup_lr(team_A_name, team_B_name, model, data, feature_cols):
    # Helper normalization to allow case-insensitive matching.
    def normalize_name(name):
        return name.strip().lower()

    team_A_norm = normalize_name(team_A_name)
    team_B_norm = normalize_name(team_B_name)

    # Match based on the "Mapped ESPN Team Name" column.
    team_A_data = data[data["Mapped ESPN Team Name"].str.lower().str.strip() == team_A_norm]
    team_B_data = data[data["Mapped ESPN Team Name"].str.lower().str.strip() == team_B_norm]

    if team_A_data.empty or team_B_data.empty:
        raise ValueError("One or both team names not found in the data.")

    # For current season data, each team should have one record.
    team_A_features = team_A_data[feature_cols].values[0].reshape(1, -1)
    team_B_features = team_B_data[feature_cols].values[0].reshape(1, -1)

    prob_A = model.predict_proba(team_A_features)[0][1]  # Probability for class 1
    prob_B = model.predict_proba(team_B_features)[0][1]

    # Calculate win probability based on relative probabilities.
    win_probability_A = prob_A / (prob_A + prob_B)
    print(f"\n{team_A_name} win probability: {win_probability_A:.4f}")
    print(f"{team_B_name} win probability: {1 - win_probability_A:.4f}")

    predicted_winner = team_A_name if win_probability_A >= 0.5 else team_B_name
    return predicted_winner

while True:
    team_A_input = input("\nEnter the name of Team A (as in 'Mapped ESPN Team Name'): ").strip()
    if team_A_input.lower() == "quit":
        break
    team_B_input = input("Enter the name of Team B: ").strip()
    if team_B_input.lower() == "quit":
        break
    try:
        winner = predict_matchup_lr(team_A_input, team_B_input, log_reg, df, feature_cols)
        print("Predicted Winner:", winner)
    except ValueError as e:
        print("Error:", e)
