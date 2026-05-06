import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# --- Data ---
iris = load_iris()
X, y = iris.data, iris.target  # (150, 4) features, 3 classes

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

# --- Model ---
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x  # raw logits — CrossEntropyLoss expects these

model     = SimpleClassifier(input_size=4, hidden_size=16, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- Training ---
EPOCHS = 100

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:3d}/{EPOCHS}  loss: {avg_loss:.4f}")

# --- Save ---
torch.save(model.state_dict(), "model.pth")
joblib.dump(scaler, "scaler.pkl")
print("\nModel saved to model.pth")
print("Scaler saved to scaler.pkl")

# --- Test accuracy ---
model.eval()
with torch.no_grad():
    logits = model(X_test)
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == y_test).float().mean().item()

print(f"Test accuracy: {accuracy * 100:.1f}%  ({(predictions == y_test).sum().item()}/{len(y_test)} correct)")
