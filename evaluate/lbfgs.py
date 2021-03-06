import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# encoding for spirograph train set
def encode_train_set_transformed(clftrainloader, device, net, col_distort, batch_transform):
    net.eval()

    store = []
    with torch.no_grad():
        t = tqdm(enumerate(clftrainloader), desc='Encoded: **/** ', total=len(clftrainloader),bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (inputs, targets) in t:
            inputs, targets = inputs.to(device), targets.to(device)
            rn1 = col_distort.sample_random_numbers(inputs.shape, inputs.device)
            inputs = batch_transform(inputs, rn1)
            representation = net(inputs)
            store.append((representation, targets))

            t.set_description('Encoded %d/%d' % (batch_idx, len(clftrainloader)))

    X, y = zip(*store)
    X, y = torch.cat(X, dim=0), torch.cat(y, dim=0)
    return X, y


def encode_train_set(clftrainloader, device, net):
    net.eval()

    store = []
    with torch.no_grad():
        t = tqdm(enumerate(clftrainloader), desc='Encoded: **/** ', total=len(clftrainloader),
                 bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (inputs, targets) in t:
            inputs, targets = inputs.to(device), targets.to(device)
            representation = net(inputs)
            store.append((representation, targets))

            t.set_description('Encoded %d/%d' % (batch_idx, len(clftrainloader)))

    X, y = zip(*store)
    X, y = torch.cat(X, dim=0), torch.cat(y, dim=0)
    return X, y


def train_clf(X, y, representation_dim, num_classes, device, reg_weight=1e-3, n_lbfgs_steps=500):
    print('\nL2 Regularization weight: %g' % reg_weight)

    criterion = nn.CrossEntropyLoss()

    # Should be reset after each epoch for a completely independent evaluation
    clf = nn.Linear(representation_dim, num_classes).to(device)
    clf_optimizer = optim.LBFGS(clf.parameters())
    clf.train()

    t = tqdm(range(n_lbfgs_steps), desc='Loss: **** | Train Acc: ****% ', bar_format='{desc}{bar}{r_bar}')
    for _ in t:
        def closure():
            clf_optimizer.zero_grad()
            raw_scores = clf(X)
            loss = criterion(raw_scores, y)
            loss += reg_weight * clf.weight.pow(2).sum()
            loss.backward()

            _, predicted = raw_scores.max(1)
            correct = predicted.eq(y).sum().item()

            t.set_description('Loss: %.3f | Train Acc: %.3f%% ' % (loss, 100. * correct / y.shape[0]))

            return loss

        clf_optimizer.step(closure)

    return clf


def test(testloader, device, net, clf):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    clf.eval()
    test_clf_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        t = tqdm(enumerate(testloader), total=len(testloader), desc='Loss: **** | Test Acc: ****% ',
                 bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (inputs, targets) in t:
            inputs, targets = inputs.to(device), targets.to(device)
            representation = net(inputs)
            # test_repr_loss = criterion(representation, targets)
            raw_scores = clf(representation)
            clf_loss = criterion(raw_scores, targets)

            test_clf_loss += clf_loss.item()
            _, predicted = raw_scores.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            t.set_description('Loss: %.3f | Test Acc: %.3f%% ' % (test_clf_loss / (batch_idx + 1), 100. * correct / total))

    acc = 100. * correct / total
    return acc, test_clf_loss / len(testloader)


def test_matrix(X, y, clf):
    criterion = nn.CrossEntropyLoss()
    clf.eval()
    with torch.no_grad():
        raw_scores = clf(X)
        test_clf_loss = criterion(raw_scores, y)

        _, predicted = raw_scores.max(1)
        correct = predicted.eq(y).sum().item()

    acc = 100. * correct / y.shape[0]
    print('Loss: %.3f | Test Acc: %.3f%%' % (test_clf_loss, acc))
    return acc, test_clf_loss.item()


def train_reg(X, y, device, reg_weight=1e-3, n_lbfgs_steps=500):
    print('\nL2 Regularization weight: %g' % reg_weight)

    criterion = nn.MSELoss()
    reg = nn.Linear(X.shape[-1], y.shape[-1]).to(device)
    clf_optimizer = optim.LBFGS(reg.parameters())
    reg.train()

    t = tqdm(range(n_lbfgs_steps), desc='Loss: **** | Train Acc: ****% ', bar_format='{desc}{bar}{r_bar}')
    for _ in t:
        def closure():
            clf_optimizer.zero_grad()
            raw_scores = reg(X)
            loss = criterion(raw_scores, y)
            loss += reg_weight * reg.weight.pow(2).sum()
            loss.backward()

            t.set_description('Loss: %.5f' % (loss))

            return loss

        clf_optimizer.step(closure)

    return reg


def test_reg(X, y, reg):
    criterion = nn.MSELoss()
    reg.eval()
    with torch.no_grad():
        raw_scores = reg(X)
        test_loss = criterion(raw_scores, y)

    print('Loss: %.5f' % (test_loss))
    return test_loss.item()


def test_reg_component(X, y, reg):
    criterion = nn.MSELoss()
    reg.eval()
    with torch.no_grad():
        raw_scores = reg(X)
        loss_list =[]
        for i in range(raw_scores.size()[1]):
            loss_list.append(criterion(raw_scores[:,i],y[:,i]))
    return torch.tensor(loss_list)


def encode_feature_averaging(clftrainloader, device, net, col_distort, batch_transform, target=None, num_passes=10):
    if target is None:
        target = device

    net.eval()

    X, y = [], None
    with torch.no_grad():
        for _ in tqdm(range(num_passes)):
            store = []
            for batch_idx, (inputs, targets) in enumerate(clftrainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                rn = col_distort.sample_random_numbers(inputs.shape, inputs.device)
                inputs = batch_transform(inputs, rn)
                representation = net(inputs)
                representation, targets = representation.to(target), targets.to(target)
                store.append((representation, targets))

            Xi, y = zip(*store)
            Xi, y = torch.cat(Xi, dim=0), torch.cat(y, dim=0)
            X.append(Xi)

    X = torch.stack(X, dim=0)

    return X, y
