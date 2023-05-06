import torch
import torch.nn as nn
import os
from torch import optim
from model import VisionTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from data_loader import get_loader


class Solver(object):
    def __init__(self, rank, args):
        self.args = args

        self.train_loader, self.test_loader = get_loader(args)

        self.model = VisionTransformer(rank, args).cuda()
        self.ce = nn.CrossEntropyLoss()

        print('--------Network--------')
        print(self.model)

        if args.load_model:
            print("Using pretrained model")
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, 'Transformer.pt')))

    def test_dataset(self, db='test'):
        self.model.eval()

        actual = []
        pred = []

        if db.lower() == 'train':
            loader = self.train_loader
        elif db.lower() == 'test':
            loader = self.test_loader

        loss = 0
        for (imgs, labels) in loader:
            imgs, labels = imgs.cuda(), labels.cuda()

            with torch.no_grad():
                class_out = self.model(imgs)
                test_loss = self.ce(class_out, labels)
            loss += test_loss.item()
            _, predicted = torch.max(class_out.data, 1)

            actual += labels.tolist()
            pred += predicted.tolist()

        loss = loss / len(loader)
        acc = accuracy_score(y_true=actual, y_pred=pred)
        cm = confusion_matrix(y_true=actual, y_pred=pred, labels=range(self.args.n_classes))

        return acc, cm, loss

    def test(self):
        train_acc, cm, train_loss = self.test_dataset('train')
        print(f"Tr Acc: {train_acc:.4f}, Tr Loss: {train_loss:.4f}")
        print(cm)

        test_acc, cm, test_loss = self.test_dataset('test')
        print(f"Te Acc: {test_acc:.4f}, Te Loss: {test_loss:.4f}")
        print(cm)

        return train_acc, test_acc, train_loss, test_loss

    def train(self):
        best_test_acc = 0
        best_test_loss = None
        best_train_acc = 0
        best_train_loss = None

        iter_per_epoch = len(self.train_loader)

        optimizer = optim.AdamW(self.model.parameters(), self.args.lr, weight_decay=1e-3)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs)

        for epoch in range(self.args.epochs):
            loss = 0
            actual = []
            pred = []
            self.model.train()

            for i, (imgs, labels) in enumerate(self.train_loader):

                imgs, labels = imgs.cuda(), labels.cuda()

                logits = self.model(imgs)
                clf_loss = self.ce(logits, labels)
                loss += clf_loss.item()

                _, predicted = torch.max(logits.data, 1)
                actual += labels.tolist()
                pred += predicted.tolist()

                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()

                if i % 50 == 0 or i == (iter_per_epoch - 1):
                    print('Ep: %d/%d, it: %d/%d, err: %.4f' % (
                    epoch + 1, self.args.epochs, i + 1, iter_per_epoch, clf_loss))

            train_acc = accuracy_score(y_true=actual, y_pred=pred)

            train_loss = loss / len(self.train_loader)
            print(f'Ep: {epoch + 1}/{self.args.epochs}, mean training loss: {loss:.4f}')

            # train_acc, cm, train_loss = self.test_dataset('train')
            test_acc, cm, test_loss = self.test_dataset('test')
            print("Test acc: %0.4f" % (test_acc))
            print(cm, "\n")

            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_train_loss = train_loss
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_loss = test_loss

            cos_decay.step()

        return best_train_acc, best_test_acc, best_train_loss, best_test_loss