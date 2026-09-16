# Customer Segmentation using Self-Organizing Map (SOM)

## 1. Project Overview

This project demonstrates how a Self-Organizing Map (SOM), also known as a Kohonen Map, can be used to group customers based on their purchasing behavior.

The project uses:
- Purchase Frequency
- Average Spending
- Product Preferences

No predefined customer categories are given to the model. The SOM learns the patterns from the customer data and maps similar customers close to each other.

## 2. Objective

The main objective is to understand how a Self-Organizing Map can be used for unsupervised customer segmentation.

The project specifically analyzes:
- How customer patterns are organized on the SOM
- The role of the winning neuron
- The role of the neighborhood around the winning neuron
- How the resulting map can be used for customer segmentation

## 3. Dataset

The dataset contains customer purchasing information.

Features used:
- Purchase_Frequency
- Average_Spending
- Electronics
- Fashion
- Grocery
- Beauty
- Sports

`Customer_ID` is used only to identify customers and is not used as an input feature for training.

## 4. Methodology

### Step 1: Load the Dataset

The customer dataset is loaded using Pandas.

### Step 2: Select Features

Customer ID is removed from the input features. The remaining purchasing attributes are used for the SOM.

### Step 3: Normalize the Data

StandardScaler is used to normalize the features so that variables with larger numerical values do not dominate the learning process.

### Step 4: Train the SOM

A 7 × 7 Self-Organizing Map is created using MiniSom.

The SOM learns customer patterns by adjusting neuron weights during training.

### Step 5: Find the Winning Neuron

For each customer, the neuron whose weight vector is most similar to the customer's feature vector is selected as the Best Matching Unit (BMU), also called the winning neuron.

### Step 6: Analyze Neighborhoods

When a neuron wins, nearby neurons are also updated during learning. This helps preserve the similarity relationships between customers on the map.

### Step 7: Customer Segmentation

The positions of the winning neurons are further grouped using K-Means clustering to create customer segments.

## 5. Results

The project produces:

- SOM distance map
- SOM customer map with customer IDs
- Customer cluster visualization
- Cluster profile summary

The cluster profile helps identify differences between customer groups based on purchasing frequency, spending, and product preferences.

## 6. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- MiniSom

## 7. Project Structure

```text
customer-som-project/
├── README.md
├── requirements.txt
├── dataset/
│   └── customer_data.csv
├── src/
│   └── som_customer_segmentation.py
├── notebooks/
│   └── SOM_Analysis.ipynb
├── results/
│   ├── som_map.png
│   ├── cluster_visualization.png
│   └── cluster_summary.csv
└── screenshots/
    └── output.png