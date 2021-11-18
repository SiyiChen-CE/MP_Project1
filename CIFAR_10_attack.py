import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10
from art.attacks.evasion import ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainer

# Load trained model
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=5, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.ReLU(),
    nn.Dropout2d(),
    nn.Conv2d(32, 32, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=3, stride=2),
    nn.Conv2d(32, 64, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(576, 64),
    nn.ReLU(),
    nn.Linear(64, 10))

saved_state_dict = torch.load('CIFAR_10.pt')
model.load_state_dict(saved_state_dict)

# Load test dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
x_train = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

train_dataset = torchvision.datasets.CIFAR10(
    './data', train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

test_dataset = torchvision.datasets.CIFAR10(
    './data', train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

classifier = PyTorchClassifier( model = model,
                                loss = nn.CrossEntropyLoss(),
                                optimizer = optim.Adam(model.parameters(), lr=0.002),
                                input_shape = (3,32,32),
                                nb_classes = 10)

attack = FastGradientMethod(estimator=classifier, eps=0.2)

# Evaluate the ART classifier on benign test examples
# correct = 0
# with torch.no_grad():
#     for data, target in test_loader:
#         output = model(data)
#         pred = output.argmax(dim=1)
#         correct += (pred == target).long().sum().item()
#
# print("Accuracy on benign test examples: {}%".format(100. * correct / len(test_loader.dataset)))

# Generate adversarial test examples
# x_test_adv = attack.generate(x=x_test)
pgd = ProjectedGradientDescent(classifier, eps=8 / 255, eps_step=2 / 255, max_iter=10, num_random_init=1)
x_test_pgd = pgd.generate(x_test)

# Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_pgd)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

# Create adversarial trainer and perform adversarial training
adv_trainer = AdversarialTrainer(classifier, attacks=pgd, ratio=1.0)
# adv_trainer.fit_generator(train_loader, nb_epochs=10)

# Evaluate the adversarially trained model on clean test set
labels_true = np.argmax(y_test, axis=1)
labels_test = np.argmax(classifier.predict(x_test), axis=1)
print("Accuracy test set: %.2f%%" % (np.sum(labels_test == labels_true) / x_test.shape[0] * 100))

# Evaluate the adversarially trained model on original adversarial samples
labels_pgd = np.argmax(classifier.predict(x_test_pgd), axis=1)
print(
    "Accuracy on original PGD adversarial samples: %.2f%%" % (np.sum(labels_pgd == labels_true) / x_test.shape[0] * 100)
)

# Evaluate the adversarially trained model on fresh adversarial samples produced on the adversarially trained model
x_test_pgd = pgd.generate(x_test)
labels_pgd = np.argmax(classifier.predict(x_test_pgd), axis=1)
print("Accuracy on new PGD adversarial samples: %.2f%%" % (np.sum(labels_pgd == labels_true) / x_test.shape[0] * 100))