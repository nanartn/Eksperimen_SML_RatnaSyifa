
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset_raw/creditcard.csv")

df = df.dropna()

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train["Amount"] = scaler.fit_transform(
    X_train[["Amount"]]
)

X_test["Amount"] = scaler.transform(
    X_test[["Amount"]]
)

X_train["Time"] = scaler.fit_transform(
    X_train[["Time"]]
)

X_test["Time"] = scaler.transform(
    X_test[["Time"]]
)

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

train.to_csv("train_preprocessing.csv", index=False)
test.to_csv("test_preprocessing.csv", index=False)

print("Preprocessing selesai.")
