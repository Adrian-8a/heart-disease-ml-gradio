from statistics import mode

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")

df = pd.read_csv("heart.csv").drop_duplicates(keep="first")
y = df["output"]
X = df.drop(["output"], axis=1)

scaler = StandardScaler()
X_std = scaler.fit_transform(X)


def cluster_to_class(clusters):
    mapping = {}
    for c in np.unique(clusters):
        mapping[c] = y[clusters == c].mode()[0]
    return mapping


models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=2),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(max_depth=10, random_state=42),
}
for model in models.values():
    model.fit(X_std, y)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_std)
kmeans_map = cluster_to_class(kmeans.labels_)

cntr, u, *_ = fuzz.cluster.cmeans(X_std.T, 2, 2, error=0.005, maxiter=1000, seed=42)
fuzzy_map = cluster_to_class(np.argmax(u, axis=0))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

LABELS = {0: "Higher risk", 1: "Lower risk"}


def predict(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall):
    patient = pd.DataFrame([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]],
                           columns=X.columns)
    x = scaler.transform(patient)

    votes = {name: int(model.predict(x)[0]) for name, model in models.items()}
    votes["KMeans"] = int(kmeans_map[kmeans.predict(x)[0]])
    u_new, *_ = fuzz.cluster.cmeans_predict(x.T, cntr, 2, error=0.005, maxiter=1000, seed=42)
    votes["Fuzzy c-means"] = int(fuzzy_map[np.argmax(u_new, axis=0)[0]])

    final = mode(votes.values())
    plt.close("all")
    table = pd.DataFrame({"Model": list(votes), "Prediction": [LABELS[v] for v in votes.values()]})

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.Spectral, s=10)
    point = pca.transform(x)
    ax.scatter(point[:, 0], point[:, 1], c="g", s=80, marker="*", label="Patient")
    ax.set_title(f"PCA, Exp. Variance: {np.round(np.sum(pca.explained_variance_ratio_), 4)}")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()

    return f"{LABELS[final]} ({list(votes.values()).count(final)}/7 models)", table, fig


inputs = [
    gr.Slider(18, 100, value=63, step=1, label="Age"),
    gr.Radio([("Female", 0), ("Male", 1)], value=1, label="Sex"),
    gr.Dropdown([("Asymptomatic", 0), ("Atypical angina", 1), ("Non-anginal pain", 2), ("Typical angina", 3)],
                value=3, label="Chest pain type (cp)"),
    gr.Number(value=145, label="Resting blood pressure, mm Hg (trtbps)"),
    gr.Number(value=233, label="Cholesterol, mg/dl (chol)"),
    gr.Radio([("No", 0), ("Yes", 1)], value=1, label="Fasting blood sugar > 120 mg/dl (fbs)"),
    gr.Dropdown([("Left ventricular hypertrophy", 0), ("Normal", 1), ("ST-T wave abnormality", 2)],
                value=1, label="Resting ECG (restecg)"),
    gr.Number(value=150, label="Maximum heart rate (thalachh)"),
    gr.Radio([("No", 0), ("Yes", 1)], value=0, label="Exercise induced angina (exng)"),
    gr.Number(value=2.3, label="ST depression, oldpeak"),
    gr.Dropdown([0, 1, 2], value=0, label="Slope of peak exercise ST (slp)"),
    gr.Dropdown([0, 1, 2, 3, 4], value=0, label="Number of major vessels (caa)"),
    gr.Dropdown([0, 1, 2, 3], value=1, label="Thalassemia (thall)"),
]

outputs = [
    gr.Textbox(label="Majority vote"),
    gr.Dataframe(label="Prediction by model"),
    gr.Plot(label="Patient on the PCA projection"),
]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title="Heart Disease Prediction",
    description="7 machine learning models vote on the risk of heart disease. Educational project, not a medical tool.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
