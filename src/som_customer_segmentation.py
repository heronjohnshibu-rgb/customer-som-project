import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("dataset/customer_data.csv")


# Select features
features = data.drop(columns=["Customer_ID"])

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(features)

# Create SOM
som = MiniSom(7, 7, X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)

# Train SOM
som.train_random(X, 1000)

# Find winning neurons
winners = [som.winner(x) for x in X]

# Convert neuron positions
positions = [[x[0], x[1]] for x in winners]

# Cluster customers based on SOM neurons
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(positions)

data["Cluster"] = clusters

# Analyze each customer segment
cluster_summary = data.groupby("Cluster").agg({
    "Purchase_Frequency": "mean",
    "Average_Spending": "mean",
    "Electronics": "mean",
    "Fashion": "mean",
    "Grocery": "mean",
    "Beauty": "mean",
    "Sports": "mean"
}).round(2)

print("\nCluster Profile:\n")
print(cluster_summary)
cluster_summary.to_csv("results/cluster_summary.csv")

# Print results
print("\nCustomer Segmentation Results:\n")
print(data[["Customer_ID", "Cluster"]].head(20))

# SOM distance map
plt.figure(figsize=(8, 6))

plt.imshow(som.distance_map().T, cmap="viridis")
plt.colorbar(label="Neuron Distance")

for i, (x, y) in enumerate(winners):
    plt.text(
        x,
        y,
        data.iloc[i]["Customer_ID"],
        ha="center",
        va="center",
        fontsize=7
    )

plt.title("SOM Customer Map with Winning Neurons")
plt.xlabel("SOM X Neuron")
plt.ylabel("SOM Y Neuron")

plt.savefig("results/som_map.png")
plt.show()

# Cluster visualization
plt.figure(figsize=(8, 6))

for cluster in range(4):
    points = [positions[i] for i in range(len(positions))
              if clusters[i] == cluster]

    x = [p[0] for p in points]
    y = [p[1] for p in points]

    plt.scatter(x, y, label=f"Cluster {cluster}")

plt.title("Customer Segmentation using SOM")
plt.xlabel("SOM X Position")
plt.ylabel("SOM Y Position")
plt.legend()
plt.savefig("results/cluster_visualization.png")
plt.show()