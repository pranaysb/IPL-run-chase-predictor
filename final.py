import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

# USER INPUT
TARGET_RUNS = int(input("Enter target runs (e.g., 25): "))
TARGET_BALLS = int(input("Enter balls left (e.g., 12): "))

# LOAD DATA
df = pd.read_csv('deliveries.csv')
#check from last 5 years only
df = df[df['match_id'] >= 113656]

# and considers only from 15+ overs
df = df[df['over'] >= 15]

# and consider only 100+ 
bat_counts = df.groupby('batter')['ball'].count()
valid_batters = bat_counts[bat_counts >= 100].index
df = df[df['batter'].isin(valid_batters)]

# strike rate between 16-20
clutch_df = df[df['over'].between(16, 20)]
clutch_stats = clutch_df.groupby('batter').agg({'batsman_runs': 'sum', 'ball': 'count'}).reset_index()
clutch_stats['clutch_sr'] = clutch_stats['batsman_runs'] / clutch_stats['ball'] * 100

# overall strike rate calculation
overall_stats = df.groupby('batter').agg({'batsman_runs': 'sum', 'ball': 'count'}).reset_index()
overall_stats['strike_rate'] = overall_stats['batsman_runs'] / overall_stats['ball'] * 100

# Normalize both SRs
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

clutch_stats['clutch_sr_norm'] = normalize(clutch_stats['clutch_sr'])
overall_stats['strike_rate_norm'] = normalize(overall_stats['strike_rate'])

# Merge and compute final clutch score
stats = pd.merge(overall_stats[['batter', 'strike_rate_norm']], clutch_stats[['batter', 'clutch_sr_norm']], on='batter', how='left').fillna(0)
stats['clutch_score'] = 0.4 * stats['strike_rate_norm'] + 0.6 * stats['clutch_sr_norm']
clutch_score_dict = dict(zip(stats['batter'], stats['clutch_score']))

# PREPARE TRAINING DATA
samples = []
labels = []
target_runs_list = []
target_balls_list = []

for (match_id, batter), group in df.groupby(['match_id', 'batter']):
    group = group.reset_index(drop=True)
    if len(group) < TARGET_BALLS:
        continue
    for i in range(len(group) - TARGET_BALLS + 1):
        runs = group.loc[i:i+TARGET_BALLS-1, 'batsman_runs'].sum()
        label = 1 if runs >= TARGET_RUNS else 0
        samples.append(batter)
        labels.append(label)
        target_runs_list.append(TARGET_RUNS)
        target_balls_list.append(TARGET_BALLS)

if len(samples) == 0:
    print("Not enough samples to train the model. Try reducing the target runs or balls.")
    exit()

# encoding the batters
le = LabelEncoder()
batter_encoded = le.fit_transform(samples)

X = np.column_stack((batter_encoded, target_runs_list, target_balls_list))
y = np.array(labels)

# splitting train test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model building
input_batter = Input(shape=(1,))
input_runs = Input(shape=(1,))
input_balls = Input(shape=(1,))

embedding = Embedding(input_dim=len(le.classes_), output_dim=10)(input_batter)
flat = Flatten()(embedding)

concat = Concatenate()([flat, input_runs, input_balls])
dense1 = Dense(16, activation='relu')(concat)
output = Dense(1, activation='sigmoid')(dense1)

model = Model(inputs=[input_batter, input_runs, input_balls], outputs=output)
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# we will train  model here 
model.fit([X_train[:,0], X_train[:,1], X_train[:,2]], y_train,
          epochs=10, batch_size=32,
          validation_data=([X_test[:,0], X_test[:,1], X_test[:,2]], y_test))

#  prediction of batsmen
all_batters = np.arange(len(le.classes_))
target_runs_array = np.full_like(all_batters, TARGET_RUNS)
target_balls_array = np.full_like(all_batters, TARGET_BALLS)

probs = model.predict([all_batters, target_runs_array, target_balls_array]).flatten()
batters = le.inverse_transform(all_batters)

clutch_scores = np.array([clutch_score_dict.get(b, 0.3) for b in batters])
final_scores = probs * 0.5 + clutch_scores * 0.5

# TOP 5 FINISHERS
top5_idx = np.argsort(final_scores)[::-1][:5]

print(f"\nTop 5 predicted clutch finishers likely to score {TARGET_RUNS} in {TARGET_BALLS} balls:")
for i, idx in enumerate(top5_idx, 1):
    print(f"{i}. {batters[idx]} - Model Prob: {probs[idx]:.2f}, Clutch Score: {clutch_scores[idx]:.2f}, Final Score: {final_scores[idx]:.2f}")
