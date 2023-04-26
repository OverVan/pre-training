import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from model.model import make
from dataset.simple_dataset import SimpleDataset
from utils import log, fix_seed, make_optimizer, make_lr_scheduler

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    fix_seed()
    
    config_path = "config/train_cifar-fs_convnet4.yaml"
    with open(config_path, "r", encoding="UTF-8") as file:
        config = yaml.load(file, yaml.FullLoader)
        
    save_dir = os.path.join("save", "{}_{}".format(config["train_dataset"]["name"], config["model"]["args"]["encoder"]["name"]))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    max_path = os.path.join(save_dir, "max-val.pth")
    last_path = os.path.join(save_dir, "last-epoch.pth")
            
    train_dataset = SimpleDataset(**config["train_dataset"]["args"])
    train_loader = DataLoader(
        train_dataset,
        config["batch_size"],
        True,
        num_workers=8,
        pin_memory=True
    )
    val_dataset = SimpleDataset(**config["val_dataset"]["args"])
    val_loader = DataLoader(
        val_dataset,
        config["batch_size"],
        True,
        num_workers=8,
        pin_memory=True
    )
    
    model = make(config["model"]["name"], **config["model"]["args"])
    criterion = CrossEntropyLoss()
    # 接续训练
    if os.path.exists(last_path):
        checkpoint = torch.load(last_path)
        start_epoch = checkpoint["last_epoch"]
        log("continue to train from epoch {}".format(start_epoch + 1))
        model.load_state_dict(checkpoint["model_sd"])
        optimizer = make_optimizer(config["optimizer"]["name"], model.parameters(), **config["optimizer"]["args"])
        optimizer.load_state_dict(checkpoint["optimizer_sd"])
        # 位置实参一定在关键字实参前面，这跟形参是一样的
        lr_scheduler = make_lr_scheduler(config["lr_scheduler"]["name"], optimizer, **config["lr_scheduler"]["args"], last_epoch=checkpoint["last_epoch"] - 1)
        max_acc = checkpoint["max_acc"]
        best_epoch = checkpoint["best_epoch"]
    else:
        log("pre-train {} {}".format(config["train_dataset"]["name"], config["model"]["args"]["encoder"]["name"]))
        start_epoch = 0
        max_acc = 0
        best_epoch = 0
        optimizer = make_optimizer(config["optimizer"]["name"], model.parameters(), **config["optimizer"]["args"])
        lr_scheduler = make_lr_scheduler(config["lr_scheduler"]["name"], optimizer, **config["lr_scheduler"]["args"])
    
    max_epoch = config["max_epoch"]
    
    for epoch in range(start_epoch, max_epoch):
        epoch_id = epoch + 1
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        
        model.train()
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            loss = criterion(pred, y)
            acc = (torch.argmax(pred, dim=1) == y).float().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
            
        lr_scheduler.step()
        
        model.eval()
        for x, y in val_loader:
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
                acc = (torch.argmax(pred, dim=1) == y).float().mean()
            val_loss.append(loss.item())
            val_acc.append(acc.item())
            
        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)
        val_loss = np.mean(val_loss)
        val_acc = np.mean(val_acc)
        
        log("epoch {}:\n\ttrain_loss: {:.4f}\ttrain_acc: {:.2%}\n\tval_loss: {:.4f}\tval_acc: {:.2%}".format(epoch_id, train_loss, train_acc, val_loss, val_acc))
        save_path = os.path.join(save_dir, "epoch-{}.pth".format(epoch_id))
        # if epoch_id > 80:
        #     torch.save(model.state_dict(), save_path)
        if val_acc >= max_acc:
            max_acc = val_acc
            best_epoch = epoch_id
            torch.save(model.state_dict(), max_path)
        log("\tbest epoch: {}\tmax acc: {:.2%}".format(best_epoch, max_acc))
        checkpoint = {
            "last_epoch": epoch_id,
            "model_sd": model.state_dict(),
            "optimizer_sd": optimizer.state_dict(),
            "max_acc": max_acc,
            "best_epoch": best_epoch
        }
        torch.save(checkpoint, last_path)