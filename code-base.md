# Deep Learning: Code Base

## 1. Perceptron Learning
### Numpy Basics
```
numpy.ones((X.shape[0], 1))
numpy.negative(numpy.ones(size))
numpy.vstack((a, b))
numpy.hstack((a, b))
numpy.all(numpy.dot(X, w) * T > 0)
numpy.any(y_pred == pred)
numpy.abs(a - b)
numpy.linspace(start=-8, stop=8)
numpy.argmax(X)  # returns index
numpy.max(X)  # returns max value
numpy.atleast_2d(x)
x.flatten()
```
### Indexing Numpy Arrays
```
X[:, 1:]  # excludes first column
X[1:, :]  # excludes first row
```
### Extract from Normal Distribution
$\mu=(-5,3)^\top$ and $\sigma=(1, 1)^\top$
```
numpy.random.normal(loc=mu, scale=sigma, size=(number_of_samples, len_of_vector))
```
### Numpy Least Squares
```
numpy.linalg.lstsq(X, T, rcond=None)[0]
```
### Sign Function
```
numpy.sign(X)
1 if numpy.dot(x, w) >= 0 else -1
```
### Random Weight Initialization
$\vec{w}\in[-10,10]^2$
```
numpy.random.uniform(low=-10, high=10, size=2)
```
### Line Parameters
$x_2=\beta x_1+\gamma$
```
beta = -w[1]/w[2]
gamma = -w[0]/w[2]

x_2 = beta * x_1 + gamma
```
### Conditional Plotting
```
pyplot.plot(X[T > 0, 1], X[T > 0, 2], "g.", label="positive data")
```

## 2. Gradient Descent (w/o Network)
```
def get_updated_weights(w, eta, grad):
  return w - eta * grad, w.copy()

def gradient_descent(w, eta=0.01, tolerance_value=1e-8, num_iterations=1000):
  w_star = w.copy()
  counter = 0

  w_star, w_old = get_updated_weights(w_star, eta, gradient(w_star))
  loss_difference = loss(w_star) - loss(w_old)

  while counter < num_iterations and \
        loss_difference > tolerance_value and \
        numpy.linalg.norm(gradient(w_star)) > tolerance_value:
        
    counter += 1
    
    w_star, w_old = get_updated_weights(w_star, eta, gradient(w_star))
    loss_difference = loss(w_star) - loss(w_old)

  return w_star
```
## 3. Non-Linear Regression
### Network Implementation Vector
```
def network(x, Theta):
  W1, w2 = Theta
  a_ = numpy.dot(W1, x)
  h_ = logistic(a_)
  h = numpy.insert(h_, 0, 1)
  y = numpy.dot(w2, h)
  
  return y, h
```
### Gradient Implementation Vector
```
def gradient(X, Theta):
  W1, w2 = Theta
  dW1 = numpy.zeros((w2.shape[0], W1.shape[1]))  # dimensions (K+1, D+1)
  dw2 = numpy.zeros(w2.shape[0])
  N = len(X)

  for x, t in X:
    y, h = network(x, Theta)
    loss = y - t
    dW1 += numpy.outer(loss * w2 * h * (1 - h), x)
    dw2 += loss * h

  dW1 = dW1 * 2 / N
  dw2 = dw2 * 2 / N

  return dW1[1:, :], dw2  # ignore W1_0 first column
```
### Gradient Descent
```
def gradient_descent(X, Theta, eta):
  epochs = 10000
  W1, w2 = Theta
  
  for _ in tqdm(range(epochs)):
    dW1, dw2 = gradient(X, Theta)
    W1 -= eta * dW1
    w2 -= eta * dw2

  return W1, w2
```
### Theta Initialization
```
def initialize_theta(k, d, lower=-1, upper=1):
  return (
    numpy.random.uniform(lower, upper, (k, d + 1)),
    numpy.random.uniform(lower, upper, k + 1)
  )
```

## 4. Multi-Output Networks
### Min-Max Normalization
```
min_val = numpy.min(X_orig[1:, :], axis=1)
max_val = numpy.max(X_orig[1:, :], axis=1)

def normalize(x, min_val, max_val):
  # x = x.astype(float)
  x[1:, :] = (numpy.divide(numpy.subtract(x[1:, :].T, min_val), numpy.subtract(max_val, min_val))).T
  return x

X = normalize(X_orig, min_val, max_val)
```
### Shuffle Columns of Dataset
```
permutation = numpy.random.permutation(X.shape[1])
X = X[:, permutation]
T = T[:, permutation]
```
### Batch Processing
```
for i in range(num_batches):
  end_index = batch_size * (i + 1)  # update end_index (16 * 1, 16 * 2, ...)
  yield X[:, start_index:end_index], T[:, start_index:end_index], start_of_epoch
  start_index = end_index
  start_of_epoch = False
```
### Network Implementation Matrix
```
def network(X, Theta):
  W1, W2 = Theta
  A = numpy.dot(W1, X)
  H_ = numpy.divide(1, numpy.add(1, numpy.exp(-A)))
  H = numpy.insert(H_, 0, 1, axis=0)  # adds bias row
  Y = numpy.dot(W2, H)

  return Y, H
```
### Gradient Implementation Matrix
```
def gradient(X, T, Y, H, Theta):
  W1, W2 = Theta
  B = X.shape[1]

  g1 = numpy.multiply((2/B), (numpy.dot(numpy.multiply(numpy.dot(W2.T, numpy.subtract(Y, T)), numpy.multiply(H, 1 - H)), X.T)))
  g2 = numpy.multiply((2/B), (numpy.dot(numpy.subtract(Y, T), H.T)))

  return g1[1:,:], g2
```
### Iterative Gradient Descent
```
def gradient_descent(X, T, Theta, B, eta=0.001, mu=None):
  loss_values = []
  W1, W2 = Theta
  max_epochs = 10000

  for _, (x,t,e) in tqdm(enumerate(batch(X, T, batch_size=B, epochs=max_epochs))):
    Y, H = network(x, Theta)
    if e:
      loss_values.append(loss(Y, t))
    dW1, dW2 = gradient(x, t, Y, H, Theta)
    W1_p = W1.copy()
    W2_p = W2.copy()
    W1 -= eta * dW1
    W2 -= eta * dW2
    if mu:
      W1 -= eta * dW1 + mu * numpy.subtract(W1, W1_p)
      W2 -= eta * dW2 + mu * numpy.subtract(W2, W2_p)

  return loss_values
```
### Xavier Weight Initialization
```
D = X.shape[0] - 1
O = T.shape[0]
s_D = 1 / numpy.sqrt(D)
s_K = 1 / numpy.sqrt(K)

W1 = numpy.random.uniform(-s_D, s_D, (K, D + 1))
W2 = numpy.random.uniform(-s_K, s_K, (O, K + 1))
Theta = [W1, W2]
```

## 5. Classification
### Read CSV File
```
def dataset(dataset_file="winequality-red.csv", delimiter=";"):
  data = []
  with open(dataset_file, 'r') as f:
    csv_reader = csv.reader(f, delimiter=delimiter)
    next(csv_reader)  # skip header line
    for sample in csv_reader:
      data.append([eval(x) for x in sample])  # convert str to int or float
      
  data = torch.tensor(data)
  X = data[:, :-1]

  if dataset_file == "winequality-red.csv":
    data[:, -1:] -= 3.0
    T = data[:, -1:].squeeze().long()
  else:
    T = data[:, -1:].float()

  return X, T
```
### Split Training and Validation Data
```
def split_training_data(X, T, train_percentage=0.8):
  N = X.shape[0]  # number of samples
  train_size = int(train_percentage * N)
  X_train = X[:train_size]
  T_train = T[:train_size]
  X_val = X[train_size:]
  T_val = T[train_size:]

  return X_train, T_train, X_val, T_val
```
### Standardize Data
```
def standardize(X_train, X_val):
  mean = torch.mean(X_train, dim=0)
  std = torch.std(X_train, dim=0)
  get_std = lambda X: (X - mean) / std
  X_train_std = get_std(X_train)
  X_val_std = get_std(X_val)
  
  return X_train_std, X_val_std
```
### Network
```
def Network(D, K, O):
  return torch.nn.Sequential(
    torch.nn.Linear(D, K),  # input D -> hidden K
    torch.nn.Tanh(),  # activation
    torch.nn.Linear(K, O)  # hidden K -> output O
  )

Network(X.shape[1], 30, len(T.unique()))
```
### Accuracy Computation
```
def accuracy(Z, T):
  N = T.shape[0]
  if Z.shape[1] == 1:  # binary classification
    return torch.mean((T == (Z >= 0)).float())
  else:
    return torch.mean((T == torch.argmax(Z, dim=1)).float())  # categorical classification
```
### Training Loop
```
def train(X_train, T_train, X_val, T_val, loss, network, learning_rate=0.01, mu=0, epochs=1000):
  optimizer = torch.optim.SGD(
    params=network.parameters(),
    lr=learning_rate,
    momentum=mu
  )
  train_loss, train_acc, val_loss, val_acc = [], [], [], []

  for _ in tqdm(range(epochs)):
    optimizer.zero_grad()
    Z = network(X_train)
    J = loss(Z, T_train)
    J.backward()
    optimizer.step()
    train_loss.append(J.item())
    train_acc.append(accuracy(Z, T_train))
    
    with torch.no_grad():
      Z_val = network(X_val)
      J_val = loss(Z_val, T_val)
      val_loss.append(J_val.item())
      val_acc.append(accuracy(Z_val, T_val))

  return train_loss, train_acc, val_loss, val_acc
```
### Loss Functions
```
torch.nn.BCEWithLogitsLoss()
torch.nn.CrossEntropyLoss()
torch.nn.MSELoss()
```
### Optimizers
```
optimizer = torch.optim.SGD(
  params=network.parameters(),
  lr=eta,
  momentum=momentum
)
optimizer = torch.optim.Adam(
  params=network.parameters(),
  lr=eta
)
```

## 6. Convolutional Networks
### Enable CUDA
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
### FashionMNIST Dataset
```
def datasets(transform):
  trainset = torchvision.datasets.FashionMNIST(root="/temp/FashionMNIST", train=True, download=True, transform=transform)
  testset = torchvision.datasets.FashionMNIST(root="/temp/FashionMNIST", train=False, download=True, transform=transform)

  return trainset, testset
  
transform = torchvision.transforms.ToTensor()
trainset, testset = datasets(transform=transform)
B = 512
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=B)
testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=B//2)
```
### Convolutional Output
**Parameters**
- $I$ stands for the input dimensions
- $K$ stands for the size of the kernel
- $P$ stands for the padding
- $S$ stands for the stride, and
- $Q$ stands for the number of channels

**Convolutional Output Dimension Formula:**
$\left(\frac{I - K + 2P}{S}\right) + 1 \times Q$

**Pooling Output Dimension Formula:**
$\left(\frac{I - K}{S}\right) + 1 \times Q$

**Computation**\
First Convolution:\
$I = 28, K = 7, P = 0, S = 1, Q = Q_1 \Longrightarrow \left(\frac{28 - 7 + 2 \cdot 0}{1}\right) + 1 \times Q_1 = 22 \times 22 \times Q_1$\
First Pooling:\
$I = 22, K = 2, S = 2, Q = Q_1 \Longrightarrow \left(\frac{22 - 2}{2}\right) + 1 \times Q_1 = 11 \times 11 \times Q_1$\
Second Convolution:\
$I = 11, K = 5, P = 2, S = 1, Q = Q_2 \Longrightarrow \left(\frac{11 - 5 + 2 \cdot 2}{1}\right) + 1 \times Q_2 = 11 \times 11 \times Q_2$\
Second Pooling:\
$I = 11, K = 2, S = 2, Q = Q_2 \Longrightarrow \left(\frac{11 - 2}{2}\right) + 1 \times Q_2$

It's common practice to round down the result to ensure that the output feature map has consistent dimensions and that no information is lost. Since Since $\frac{9}{2} + 1 = 5.5$ given a number $Q_2$ of output channels $5 \times 5 \times Q_2$ hidden neurons are necessary.
### Network
```
def fully_connected(D, K1, K2, O):
  return torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(D, K1),
    torch.nn.Sigmoid(),
    torch.nn.Linear(K1, K2),
    torch.nn.Sigmoid(),
    torch.nn.Linear(K2, O)
  )
def convolutional(C, Q1, Q2, O):
  return torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=C, out_channels=Q1, kernel_size=(7, 7), stride=1, padding=0),
    torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
    torch.nn.Sigmoid(),
    torch.nn.Conv2d(in_channels=Q1, out_channels=Q2, kernel_size=(5, 5), stride=1, padding=2),
    torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
    torch.nn.Sigmoid(),
    torch.nn.Flatten(),
    torch.nn.Linear((5**2) * Q2, O)
  )
```
### Learnable Parameters
**Fully-connected Network**

For every fully-connected layer the weights $(I \cdot O)$ and the biases $(O)$ are added together, where $I$ stands for the number of input and $O$ for the number of output features.
- first fully-connected layer: $D \cdot K_1 + K_1 = 28^2 \cdot 128 + 128 = 100'480$
- second fully-connected layer: $K_1 \cdot K_2 + K_2 = 128 \cdot 64 + 64 = 8'256$
- third fully-connected layer: $K_2 \cdot O + O = 64 \cdot 10 + 10 = 650$
- total: $100'480 + 8'256 + 650 = 109'386$

**Convolutional Network**

For every convolutional layer the weights $(C \cdot O \cdot K)$ and the biases $(O)$ are added together, where $C$ stands for the number of input, $O$ for the number of output channels and $K$ for the kernel size. For the fully-conntected layer $I = K \cdot C$.
- first convolutional layer: $C \cdot Q_1 \cdot K + Q_1 = 1 \cdot 16 \cdot 7^2 + 16 = 800$
- second convolutional layer: $Q_1 \cdot Q_2 \cdot K + Q_2 = 16 \cdot 16 \cdot 5^2 + 16 = 6'416$
- fully-connected layer: $K \cdot Q_2 \cdot O + O = 5^2 \cdot 16 \cdot 10 + 10 = 4'010$
- total: $800 + 6'416 + 4'010 = 11'226$
```
def parameter_count(network):
  return sum(p.numel() for p in network.parameters())
```

## 7. Transfer Learning
### Data Transformation
```
imagenet_transform = torchvision.transforms.Compose([
  torchvision.transforms.Resize(256),
  torchvision.transforms.CenterCrop(224),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
```
### Dataset
```
from torchvision.datasets import ImageFolder

#Path to your training and test data (If different, change the path accordingly)
train_dir = './intel-image-classification/seg_train/seg_train/'
test_dir = './intel-image-classification/seg_test/seg_test/'

trainset = ImageFolder(root=train_dir, transform=imagenet_transform)
testset = ImageFolder(root=test_dir, transform=imagenet_transform)
B = 32
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True,batch_size=B)
testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=B)
```
### Pre-trained Network Instantiation
```
network_1 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
for param in network_1.parameters():
  param.requires_grad = False  # freeze layers of network

network_2 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
```
### Layer Replacement
```
def replace_last_layer(network, O=6):
  in_features = network.fc.in_features
  new_fc = torch.nn.Linear(in_features, O)
  network.fc = new_fc

  return network
```
### Training Loop
```
def train_eval(model, dataloaders, epochs=1000, eta=0.01, momentum=0):
  trainloader, testloader = dataloaders
  optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=eta,
    momentum=momentum
  )
  loss = torch.nn.CrossEntropyLoss()
  device = torch.device("cuda")
  model = model.to(device)

  train_loss, train_acc, val_loss, val_acc, curr_pred, curr_target = [], [], [], [], [], []

  for epoch in tqdm(range(epochs), desc='epoch'):
    model.train()
    train_loss_epoch = 0
    train_number_correct_pred = 0
    for x, t in tqdm(trainloader, desc='training', colour='purple', leave=False):
      optimizer.zero_grad()
      x, t = x.to(device), t.to(device)
      z = model(x)
      J = loss(z, t)
      J.backward()
      optimizer.step()
      train_loss_epoch += J.item() * trainloader.batch_size
      
      train_number_correct_pred += (torch.argmax(z, dim=1) == t).sum().item()

    train_loss.append(train_loss_epoch / len(trainloader.dataset))
    train_acc.append(train_number_correct_pred / len(trainloader.dataset))

    model.eval()
    with torch.no_grad():
      val_loss_epoch = 0
      val_number_correct_pred = 0
      for x_val, t_val in tqdm(testloader, desc='testing', colour='orange', leave=False):
        x_val, t_val = x_val.to(device), t_val.to(device)
        z_val = model(x_val)
        J_val = loss(z_val, t_val)
        val_loss_epoch += J_val.item() * testloader.batch_size
        val_number_correct_pred += (torch.argmax(z_val, dim=1) == t_val).sum().item()

        if epoch == epochs - 1:
          curr_pred.append(z_val.detach().cpu().numpy())
          curr_target.append(t_val.detach().cpu().numpy())

      val_loss.append(val_loss_epoch / len(testloader.dataset))
      val_acc.append(val_number_correct_pred / len(testloader.dataset))

    print(f"Epoch {epoch + 1}/{epochs}:")
    print(f"  Training Loss: {train_loss[-1]:.4f}")
    print(f"  Training Accuracy: {train_acc[-1]:.4f}")
    print(f"  Validation Loss: {val_loss[-1]:.4f}")
    print(f"  Validation Accuracy: {val_acc[-1]:.4f}")

  pred, target = numpy.concatenate(curr_pred), numpy.concatenate(curr_target)
  return pred, target
  
# use numpy.argmax(pred, axis=1) for average predictions
```

## 8. Open-Set Classification
### Target Vectors
```
known_classes = (1, 4, 5, 8)
negative_classes = (0, 2, 3, 7)
unknown_classes = (6, 9)
O = len(known_classes)

labels_known = torch.eye(O)
label_unknown = torch.full((4,), 0.25)
labels_combined = torch.full((10, 4), 0.25)
for i, idx in enumerate(known_classes):
    labels_combined[idx] = labels_known[i]

def target_vector(index):
  return labels_combined[index]
```
### MNIST Dataset
```
class DataSet(torchvision.datasets.MNIST):
  def __init__(self, purpose="train"):
    super(DataSet, self).__init__(
        root="/temp/MNIST",
        train=(purpose == "train"),
        download=True,
        transform=torchvision.transforms.ToTensor(),
        target_transform=target_vector
      )

    if purpose == "test":
      valid_classes = known_classes + unknown_classes
    else:
      valid_classes = known_classes + negative_classes

    mask = torch.tensor([sample in valid_classes for sample in self.targets])
    self.data = self.data[mask]
    self.targets = self.targets[mask]

batch_size = 256
train_set = DataSet(purpose="train")
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=batch_size)
validation_set = DataSet(purpose="validation")
validation_loader = torch.utils.data.DataLoader(validation_set, shuffle=False, batch_size=batch_size)
test_set = DataSet(purpose="test")
test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=batch_size)
```
### Utility Function
```
def split_known_unknown(batch, targets):
  known = torch.amy(targets == 1, dim=1)
  unknown = torch.all(targets == 1/O, dim=1)
  
  return batch[known], targets[known], batch[unknown]
```
### Loss Function Implementation
```
class AdaptedSoftMax(torch.autograd.Function):
  @staticmethod
  def forward(ctx, logits, targets):
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    ctx.save_for_backward(log_probs, targets)
    loss = - targets * log_probs
    return torch.sum(loss)

  @staticmethod
  def backward(ctx, result):
    log_probs, targets = ctx.saved_tensors
    y = torch.exp(log_probs)
    dJ_dz = result * (y - targets)
    return dJ_dz, None

adapted_softmax = AdaptedSoftMax.apply

def adapted_softmax_alt(logits, targets):
  loss = - torch.sum(targets * logits) + (1 / logits.size(1)) * torch.logsumexp(logits, dim=1).sum()
  return loss
```
### Confidence Evaluation
```
def confidence(logits, targets):
  softmax_probs = torch.nn.functional.softmax(logits, dim=1).to(device)
  batch_known, targets_known, batch_unknown = split_known_unknown(softmax_probs, targets)
  conf_known = torch.sum(batch_known * targets_known)
  conf_unknown = torch.sum(1 - torch.max(batch_unknown, dim=1).values + (1 / logits.size(1)))

  return conf_known + conf_unknown
```
### Network
```
class Network(torch.nn.Module):
  def __init__(self, Q1=32, Q2=32, K=20, O=4):
    super(Network,self).__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=Q1, kernel_size=(7, 7), stride=1, padding=0)
    self.conv2 = torch.nn.Conv2d(in_channels=Q1, out_channels=Q2, kernel_size=(5, 5), stride=1, padding=2)
    self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
    self.act = torch.nn.PReLU()
    self.flatten = torch.nn.Flatten()
    self.fc1 = torch.nn.Linear((5**2) * Q2, K)
    self.fc2 = torch.nn.Linear(K, O)

  def forward(self,x):
    a = self.act(self.pool(self.conv1(x)))
    a = self.act(self.pool(self.conv2(a)))
    deep_features = self.fc1(self.flatten(a))
    logits = self.fc2(deep_features)
    
    return logits, deep_features
```
### Training Loop
```
def train(network, loss_function, epochs=1000, eta=0.01, momentum=0):
  network = network.to(device)
  optimizer = torch.optim.SGD(
      params=network.parameters(),
      lr=eta,
      momentum=momentum
  )

  for epoch in tqdm(range(epochs), desc='epoch'):
    train_conf = validation_conf = 0.

    for x, t in tqdm(train_loader, desc='training', colour='purple', leave=False):
      optimizer.zero_grad()
      x, t = x.to(device), t.to(device)
      logits, _ = network.forward(x)
      J = loss_function(logits, t)
      J.backward()
      optimizer.step()

      train_conf += confidence(logits, t)

    with torch.no_grad():
      for x_val, t_val in tqdm(validation_loader, desc='validating', colour='orange', leave=False):
        x_val, t_val = x_val.to(device), t_val.to(device)
        logits_val, _ = network.forward(x_val)
        validation_conf += confidence(logits_val, t_val)
        
    print(f"\rEpoch {epoch+1}; train: {train_conf/len(train_set):1.5f}, val: {validation_conf/len(validation_set):1.5f}")

  return network
```
### Validation Loop
```
def plot_features(network):
  known, negative, unknown = [], [], []

  with torch.no_grad():
    for x, t in tqdm(validation_loader, desc='validation'):
      x, t = x.to(device), t.to(device)
      _, deep_features = network.forward(x)
      norms = torch.norm(deep_features, dim=1)
      batch_known, _, batch_unkown = split_known_unknown(norms, t)
      known.extend(batch_known.detach().cpu().numpy())
      negative.extend(batch_unkown.detach().cpu().numpy())

    for x,t in tqdm(test_loader, desc='test'):
      x, t = x.to(device), t.to(device)
      _, deep_features = network.forward(x)
      norms = torch.norm(deep_features, dim=1)
      batch_known, _, batch_unkown = split_known_unknown(norms, t)
      known.extend(batch_known.detach().cpu().numpy())
      unknown.extend(batch_unkown.detach().cpu().numpy())
```
### Classification Evaluation
```
def evaluation(network):
  zeta = 0.98
  correct = known = 0
  false = unknown = 0

  with torch.no_grad():
    for x,t in tqdm(test_loader, desc='test'):
      x, t = x.to(device), t.to(device)
      logits, _ = network.forward(x)
      softmax_probs = torch.nn.functional.softmax(logits, dim=1)
      batch_known, targets_known, batch_unkown = split_known_unknown(softmax_probs, t)

      correct += torch.sum(batch_known * targets_known >= zeta)
      known += len(batch_known)

      false += torch.sum(torch.max(batch_unkown, dim=1).values >= zeta)
      unknown += len(batch_unkown)

  print (f"CCR: {correct} of {known} = {correct/known*100:2.2f}%")
  print (f"FPR: {false} of {unknown} = {false/unknown*100:2.2f}%")


evaluation(network_adapted)
```

## 9. Auto-Encoder Network
### FashionMNIST and MNIST Dataset
```
class DatasetWithIndicator(torch.utils.data.Dataset):
  def __init__(self, dataset, type_indicator):
    self.dataset = dataset
    self.type_indicator = type_indicator

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    image, target = self.dataset[idx]
    return image, target, self.type_indicator


class MixedDataset(torch.utils.data.Dataset):
  def __init__(self, root='./data', purpose="train", transform=None, anomaly_size=2000):
    self.dataset = DatasetWithIndicator(
      dataset=torchvision.datasets.MNIST(root=root, train=purpose=="train", download=True, transform=transform),
      type_indicator=1
    )

    if purpose == "anomaly_detection":
      self.fashion_dataset = DatasetWithIndicator(
        dataset=torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform),
        type_indicator=-1
      )
      indices = torch.randperm(len(self.fashion_dataset))[:anomaly_size]  # select random samples
      self.fashion_dataset = torch.utils.data.Subset(self.fashion_dataset, indices)
      self.dataset = torch.utils.data.ConcatDataset([self.dataset, self.fashion_dataset])


  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    image, target, data_type = self.dataset[idx]
    return image, target, data_type

transform = torchvision.transforms.ToTensor()
train_dataset = MixedDataset(purpose="train", transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = MixedDataset(purpose="val", transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)
anomaly_detection_dataset = MixedDataset(purpose="anomaly_detection", transform=transform, anomaly_size=2000)
anomaly_detection_loader = torch.utils.data.DataLoader(anomaly_detection_dataset, batch_size=1000, shuffle=True)
```
### Encoder Network
```
class Encoder(torch.nn.Module):
  def __init__(self, Q1, Q2, K):
    super(Encoder,self).__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=Q1, kernel_size=(5, 5), stride=2, padding=2)
    self.conv2 = torch.nn.Conv2d(in_channels=Q1, out_channels=Q2, kernel_size=(5, 5), stride=2, padding=2)
    self.act = torch.nn.ReLU()
    self.flatten = torch.nn.Flatten()
    self.fc1 = torch.nn.Linear((7**2) * Q2, K)

  def forward(self, x):
    deep_feature = self.fc1(self.act(self.flatten(self.conv2(self.act(self.conv1(x))))))
    return deep_feature
```
### Decoder Network
```
class Decoder(torch.nn.Module):
  def __init__(self, Q1, Q2, K):
    super(Decoder,self).__init__()
    self.fc = torch.nn.Linear(K, (7**2) * Q2)
    self.deconv1 = torch.nn.ConvTranspose2d(in_channels=Q2, out_channels=Q1, kernel_size=(5, 5), stride=2, padding=2, output_padding=1)
    self.deconv2 = torch.nn.ConvTranspose2d(in_channels=Q1, out_channels=1, kernel_size=(5, 5), stride=2, padding=2, output_padding=1)
    self.act = torch.nn.ReLU()
    self.unflatten = torch.nn.Unflatten(1, (Q2, 7, 7))
    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x):
    output = self.sigmoid(self.deconv2(self.act(self.deconv1(self.unflatten(self.act(self.fc(x)))))))
    return output
```
### Auto-Encoder Network
```
class AutoEncoder(torch.nn.Module):
  def __init__(self, Q1, Q2, K):
    super(AutoEncoder,self).__init__()
    self.encoder = Encoder(Q1, Q2, K)
    self.decoder = Decoder(Q1, Q2, K)

  def forward(self,x):
    deep_feature = self.encoder(x)
    reconstructed = self.decoder(deep_feature)
    return reconstructed
```
### Training Loop
```
network = AutoEncoder(32, 32, 10).to(device)
optimizer = torch.optim.Adam(
  params=network.parameters(),
  lr=0.0005
)
loss = torch.nn.MSELoss()

for epoch in tqdm(range(10), desc='epoch'):
  train_loss = validation_loss = 0.

  for x, t, _ in tqdm(train_loader, desc='training', colour='purple', leave=False):
    optimizer.zero_grad()
    x = x.to(device)
    y = network(x)
    J = loss(y, x)
    J.backward()
    optimizer.step()

    train_loss += J.item() * train_loader.batch_size


  with torch.no_grad():
    for x_val, _, _ in tqdm(val_loader, desc='validating', colour='orange', leave=False):
      x_val = x_val.to(device)
      y_val = network(x_val)
      J_val = loss(y_val, x_val)
      validation_loss += J_val.item() * val_loader.batch_size

  print(f"\rEpoch {epoch+1}; train: {train_loss/len(train_dataset):1.5f}, val: {validation_loss/len(val_dataset):1.5f}")
```
### True Positive / Negative Rate
```
def compute_tpr_tnr(predictions, truth):
  predictions = numpy.array(predictions)
  truth = numpy.array(truth)
  tn, fp, fn, tp = confusion_matrix(y_true=truth, y_pred=predictions).ravel()
  tpr = tp / (tp + fn)
  tnr = tn / (tn + fp)

  return tpr, tnr
```
### Anomaly Evaluation
```
loss = torch.nn.MSELoss(reduction="none")

correct = 0.
predictions = []
truth_values = []

with torch.no_grad():
  for x, t, l in tqdm(anomaly_detection_loader, desc="anomaly_detection"):
    x, t, l = x.to(device), t.to(device), l.to(device)
    y = network(x)
    J = loss(y, x).squeeze(1).squeeze(1)
    J_per_sample = J.mean(dim=[1, 2])
    prediction = torch.where(J_per_sample > 0.04, torch.tensor(-1), torch.tensor(1))
    predictions += prediction.detach().cpu().tolist()
    truth_values += l.detach().cpu().tolist()
    correct += torch.sum(prediction == l).item()

acc = correct / len(anomaly_detection_loader.dataset)
tpr, tnr = compute_tpr_tnr(predictions, truth_values)

print(f"True Positive Rate: {tpr:1.4f}")
print(f"True Negative Rate: {tnr:1.4f}")
print(f"Accuracy: {acc:1.4f}")
```

## 10. Recurrent LSTM Network
### Data Loading
```
def get_data(datafile):
  data = pandas.read_csv(datafile)
  date = data["Date"].to_numpy().astype('datetime64[D]')
  price = torch.tensor(data['Close'].values)

  return date, price
```
### Split Training and Testing Data
```
def train_test_split(stock_data):
  dates, prices = stock_data
  split_date = np.datetime64('2018-01-01')
  split_index = np.where(dates == split_date)[0][0]
  train_dates, train_prices = dates[:split_index], prices[:split_index]
  test_dates, test_prices = dates[split_index:], prices[split_index:]
  train_data = (train_dates, train_prices)
  test_data = (test_dates, test_prices)

  return train_data, test_data
```
### Min-Max Normalization
```
def min_max_scaler(train_data, test_data):
  min_val = torch.min(train_data)
  max_val = torch.max(train_data)
  train_data_scaled = (train_data - min_val) / (max_val - min_val)
  test_data_scaled = (test_data - min_val) / (max_val - min_val)

  return train_data_scaled, test_data_scaled, min_val, max_val


def inverse_min_max_scaler(scaled_data, min_val, max_val):
  original_data = scaled_data * (max_val - min_val) + min_val

  return original_data
```
### Sequences
```
def create_sequences_targets(data: torch.Tensor, S):
  X, T = [], []

  for i in range(len(data) - S):
    sequence = data[i:i + S]
    target = data[i + S]
    X.append(sequence)
    T.append(target)

  return torch.stack(X), torch.stack(T)
```
### Dataset
```
class Dataset(torch.utils.data.Dataset):
  def __init__(self, data, S):
    self.X, self.T = create_sequences_targets(data, S)

  def __getitem__(self, index):
    return self.X[index].unsqueeze(1), self.T[index].unsqueeze(0)

  def __len__(self):
    return len(self.X)

S = 7
gail_train_dataset = Dataset(train_gail_scaled, S)
gail_train_dataloader = torch.utils.data.DataLoader(gail_train_dataset, batch_size=256, shuffle=True)
gail_test_dataset = Dataset(test_gail_scaled, S)
gail_test_dataloader = torch.utils.data.DataLoader(gail_test_dataset, batch_size=256, shuffle=False)
ntpc_train_dataset = Dataset(train_ntpc_scaled, S)
ntpc_train_dataloader = torch.utils.data.DataLoader(ntpc_train_dataset, batch_size=256, shuffle=True)
ntpc_test_dataset =  Dataset(test_ntpc_scaled, S)
ntpc_test_dataloader = torch.utils.data.DataLoader(ntpc_test_dataset, batch_size=256, shuffle=False)
```
### LSTM Network
```
class LSTMModel(torch.nn.Module):
  def __init__(self, D, K, O):
    super(LSTMModel,self).__init__()
    self.lstm = torch.nn.LSTM(input_size=D, hidden_size=K, batch_first=True,  dtype=torch.float64)
    self.dropout = torch.nn.Dropout(0.2)
    self.linear = torch.nn.Linear(K,O, dtype=torch.float64)

  def forward(self, x):
    lstm_out,_ = self.lstm(x)
    lstm_out = self.dropout(lstm_out)
    lstm_out = lstm_out[:, -1:]
    Z = self.linear(lstm_out)

    return Z
```
### Training Loop
```
def train(network,train_dataloader,optimizer,loss,device,epochs=50):
  network.to(device)
  for epoch in range(epochs):  
    network.train()
    train_loss = 0
    total_sample = len(train_dataloader)
    for x, t in train_dataloader:
      optimizer.zero_grad()
      x, t = x.to(device), t.to(device)
      y = network(x)
      J = loss(y, t)
      J.backward()
      optimizer.step()
      
      train_loss += J.item() * train_dataloader.batch_size
      
    print(f"\rEpoch {epoch+1}; train loss: {train_loss/total_sample:1.5f}")
```
### Testing Loop
```
def predict(network,test_dataloader):
  network.eval()
  predictions = []
  with torch.no_grad():
    for x, _ in test_dataloader:
      x = x.to(device)
      pred = network(x)
      predictions.append(pred.squeeze())

  return predictions
```

## 11. Adversarial Training
### MNIST Dataset
```
transform = torchvision.transforms.ToTensor()
train_set = torchvision.datasets.MNIST(root="/data/MNIST", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=32)
validation_set = torchvision.datasets.MNIST(root="/data/MNIST", train=False, download=True, transform=transform)
validation_loader = torch.utils.data.DataLoader(validation_set, shuffle=False, batch_size=100)
```
### Network
```
class Network(torch.nn.Module):
  def __init__(self, Q1, Q2, K, O):
    super(Network,self).__init__()
    self.build = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=1, out_channels=Q1, kernel_size=(7, 7), stride=1, padding=0),
      torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
      torch.nn.ReLU(),
      torch.nn.Conv2d(in_channels=Q1, out_channels=Q2, kernel_size=(5, 5), stride=1, padding=2),
      torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
      torch.nn.ReLU(),
      torch.nn.Flatten(),
      torch.nn.Linear((5**2) * Q2, K),
      torch.nn.Linear(K, O)
    )

  def forward(self,x):
    return self.build(x)
```
### FGS
```
def FGS(x, t, network, loss, alpha=0.3):
  x, t = x.to(device), t.to(device)
  x.requires_grad_()
  z = network(x)
  J = loss(z, t)
  J.backward()

  gradient = x.grad
  adversarial_sample = x + alpha * torch.sign(gradient)

  return torch.clamp(adversarial_sample, 0, 1)
```
### FGV
```
def FGV(x, t, network, loss, alpha=0.6):
  x, t = x.to(device), t.to(device)
  x.requires_grad_()
  z = network(x)
  J = loss(z, t)
  network.zero_grad()
  J.backward()

  gradient = x.grad
  adversarial_sample = x + alpha * (gradient / torch.amax(torch.abs(gradient), dim=[1, 2, 3], keepdim=True))

  return torch.clamp(adversarial_sample, 0, 1)
```
### Noise
```
def noise(x, alpha=0.3):
  x = x.to(device)
  noise = torch.randint(x.size(), device=device).float() * 2 - 1
  noisy_sample = x + alpha * noise

  return torch.clamp(noisy_sample, 0, 1)
```
### Training Loop
```
def training_loop(network, loss, optimizer, add_additional_samples=None, alpha=0.3):
  network = network.to(device)
  description = f'training {"default" if add_additional_samples is None else add_additional_samples}'
  for x, t in tqdm(train_loader, desc=description, colour='purple', leave=False):
    optimizer.zero_grad()
    x, t = x.to(device), t.to(device)
    z = network(x)
    J = loss(z, t)
    J.backward()
    optimizer.step()

    if add_additional_samples is not None:
      if add_additional_samples == "FGS":
        x_hat = FGS(x, t, network, loss, alpha)
      else:
        x_hat = noise(x, alpha)

      z_hat = network(x_hat)
      J = loss(z_hat, t)
      J.backward()
      optimizer.step()
```
### Validation Loop
```
def validation_loop(network, loss, add_additional_samples=None, alpha_fgs=0.3, alpha_fgv=0.6):
  network = network.to(device)
  correct_clean_count, correct_fgs_count, correct_fgv_count = 0, 0, 0
  description = f'validating {"default" if add_additional_samples is None else add_additional_samples}'
  for x, t in tqdm(validation_loader, desc=description, colour='orange', leave=False):
    with torch.no_grad():
      x, t = x.to(device), t.to(device)
      z = network(x)

      preds = torch.argmax(z, dim=1)
      correct_clean_count += (preds == t).sum().item()

    correct_indices = (preds == t)
    x_correct = x[correct_indices]
    t_correct = t[correct_indices]
    x_attack_fgs = FGS(x_correct, t_correct, network, loss, alpha_fgs)
    x_attack_fgv = FGV(x_correct, t_correct, network, loss, alpha_fgv)

    with torch.no_grad():
      z_attack_fgs = network(x_attack_fgs)
      z_attack_fgv = network(x_attack_fgv)
      correct_fgs_count += (torch.argmax(z_attack_fgs, dim=1) == t_correct).sum().item()
      correct_fgv_count += (torch.argmax(z_attack_fgv, dim=1) == t_correct).sum().item()

  clean_accuracy = correct_clean_count / len(validation_loader.dataset)
  fgs_accuracy = (correct_fgs_count / correct_clean_count) if correct_clean_count > 0 else 0
  fgv_accuracy = (correct_fgv_count / correct_clean_count) if correct_clean_count > 0 else 0
  return clean_accuracy, fgs_accuracy, fgv_accuracy
```
### Multiple Networks
```
number_networks = 3
networks = [Network(32, 64, 10, 10) for _ in range(number_networks)]
optimizers = [torch.optim.SGD(
  params=networks[i].parameters(),
  lr=0.005,
  momentum=0.8
) for i in range(number_networks)]

loss = torch.nn.CrossEntropyLoss()
alpha = 0.3

clean_accuracies = [[] for _ in range(number_networks)]
fgs_accuracies = [[] for _ in range(number_networks)]
fgv_accuracies = [[] for _ in range(number_networks)]

data_extensions = [None, "noise", "FGS"]

for epoch in tqdm(range(10), desc='epoch'):
  for idx, (network, optimizer, data_extension) in enumerate(zip(networks, optimizers, data_extensions)):
    training_loop(network, loss, optimizer, data_extension, alpha=alpha)
    clean, fgs, fgv = validation_loop(network, loss, data_extension, alpha_fgs=alpha, alpha_fgv=alpha*2)
    clean_accuracies[idx].append(clean)
    fgs_accuracies[idx].append(fgs)
    fgv_accuracies[idx].append(fgv)
```

## 12. Zero-Shot Learning
### Data Transformation
```
imagenet_transform = torchvision.transforms.Compose([
  torchvision.transforms.Resize((224, 224)),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
```
### CLIP Model
```
model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

label_tokens = processor(
  text=text_prompts,
  padding=True,
  images=None,
  return_tensors='pt'
).to(device)

model = model.to(device
label_emb = model.get_text_features(**label_tokens)
label_emb = label_emb.detach().cpu()
label_emb = label_emb / torch.norm(label_emb, dim=1, keepdim=True)
```
### Testing Loop
```
for imgs, labels in tqdm(testloader, desc='testing'):
  imgs = imgs.to(device)
  y_true.extend(labels.detach().cpu())
  img_embd = model.get_image_features(imgs)
  img_embd = img_embd.detach().cpu()
  scores = torch.matmul(img_embd, label_emb.T)
  pred = torch.argmax(scores, dim=1)
  y_pred.extend(pred)
```

## 13. EXAM: Assignment 1
### Target Vectors
Checkerboard Targets
```
def target(x):
  targets = []
  for i in range(len(x)):
    if (torch.floor(x[i][0]) % 2 and torch.floor(x[i][1]) % 2) or (torch.ceil(x[i][0]) % 2 and torch.ceil(x[i][1]) % 2) :
      targets.append(1)
    else:
      targets.append(-1)
      
  return torch.tensor(targets)
```
### Sample Distribution
```
def batch(B, device = "cpu"):
  X = torch.distributions.Uniform(-2,2).sample([B,2]).to(device)
  T = target(X)
    
  return X,T
```
### Piecewise Activation
$\mathrm{tent}(a) = \begin{cases}1-a & \text{for } 0\leq a\leq 2 \\ 1+a & \text{for } -2 \leq a < 0\\ -1 & \text{elsewhere}\end{cases}$
```
def tent(a):
  cond1 = torch.logical_and(0 <= a, a <= 2)
  cond2 = torch.logical_and(-2 <= a, a < 0)
  cond3 = torch.logical_or(a < -2, a > 2)
  h = cond1 * (1-a) + cond2 * (1+a) + cond3 * -1
    
  return h

class Tent(torch.autograd.Function):
  @staticmethod
  def forward(ctx, values, targets):
    a = tent(values)
    ctx.save_for_backward(values, targets)
    return a
    
  @staticmethod
  def backward(ctx, dJ_dH):
    values, targets = ctx.saved_tensors
    dH_dA = torch.zeros_like(values)
    cond1 = torch.logical_and(0 <= values, values <= 2)
    cond2 = torch.logical_and(-2 <= values, values < 0)
    cond3 = torch.logical_or(values < -2, values > 2)
    dH_dA[cond1] = -1
    dH_dA[cond2] = 1
    dH_dA[cond3] = 0
    dJ_dA = dJ_dH * dH_dA
    
    return dJ_dA, None
```

## 14. EXAM: Assignment 2
### Residual Block Implementation
```
class ResidualBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, hidden):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
    self.conv2 = torch.nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1)
    self.conv_adjustx = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
    self.relu = torch.nn.ReLU()
    
  def forward(self, x):
    y = self.relu(self.conv1(x))
    y = self.conv2(y)
    return self.relu(y + self.conv_adjustx(x))
```
### Residual Network
```
network = torch.nn.Sequential(
  ResidualBlock(1, 4, 1),
  torch.nn.MaxPool2d(2),
  ResidualBlock(4, 8, 1),
  torch.nn.MaxPool2d(2),
  ResidualBlock(8, 8, 1),
  torch.nn.Flatten(),
  torch.nn.Linear(7*7*8, 10)
).to(device)
```

## 15. EXAM: Assignment 3
### Pre-trained Network
```
network = torchvision.models.resnet18(pretrained=True)
network.eval()
```
### Image Preprocessing
```
transform_1 = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
])
transform_2 = torchvision.transforms.Compose([
   
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

X = transform_1(pil_image)
```
### Network Prediction
```
def predict(sample):
    output = network(transform_2(sample).unsqueeze(0))
    probs = torch.nn.functional.softmax(output, dim=1)
    return probs.argmax(), probs.max()
```
### Gradient Calculation
```
X.requires_grad_(True)
original_class, original_prob = predict(X)
print(f"Original sample: class={original_class}, probability={original_prob}")
true_class = 954 #banana
J = torch.nn.CrossEntropyLoss()(network(transform_2(X).unsqueeze(0)),torch.tensor([original_class]))
J.backward()
```
### Adversarial Sample
```
X_check = X + 0.5 * torch.div(X.grad, torch.max(X.grad))
adversarial_class, adversarial_prob = predict(X_check)
```
