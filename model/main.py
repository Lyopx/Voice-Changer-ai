import os
import time
from tqdm import tqdm
import logging

import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.append("../")
from config import model_path, parser
from evaluate import evaluate, compute_loss

from inference import save_pred_audios_in_training

from DatasetLoader import DatasetLoader
from Transformer import Transformer


def config_save():
    """Config save path"""
    if args.debug:
        args.save = os.path.join(model_path, "ckpts", "debug")
        args.epochs = 2
    else:
        args.save = os.path.join(model_path, "ckpts", args.dataset, experiment_name)

    if os.path.exists(args.save):
        os.system("rm -r {}".format(args.save))
    os.makedirs(args.save)

def config_log():
    """log save path"""
    logging_dir = os.path.join("logs", args.dataset)
    os.makedirs(logging_dir, exist_ok=True)

    if args.debug:
        logging_file = os.path.join(logging_dir, "debug.log")
    else:
        logging_file = os.path.join(logging_dir, "{}.log".format(experiment_name))

    # logging.basicConfig(
    #     filename=logging_file, encoding="utf-8", level=logging.DEBUG, filemode="w"
    # )
    logging.basicConfig(
        filename=logging_file, level=logging.INFO, filemode="w"
    )

    tensorboard_dir = os.path.join(logging_dir, experiment_name)
    if os.path.exists(tensorboard_dir):
        os.system("rm -r {}".format(tensorboard_dir))

    writer = SummaryWriter(log_dir=tensorboard_dir)
    return writer


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    args = parser.parse_args()

    # set experiment name
    if args.debug:
        experiment_name = "debug"
    else:
        is_training = "lr_{}".format(args.lr) if not args.evaluate else "eval"
        is_conversing = "" if not args.converse else "conversion"
        experiment_keys = [
            args.model,
            is_training,
            is_conversing,
        ]
        experiment_keys = [s for s in experiment_keys if s != ""]
        experiment_name = "_".join(experiment_keys)

    # Save dir
    config_save()

    # Logging
    writer = config_log()

    logging.info(
        "\n{} Experimental Dataset: {} {}\n".format("=" * 20, args.dataset, "=" * 20)
    )
    logging.info("save path: {}".format(args.save))
    logging.info(
        "Start time: {}".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        )
    )

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    logging.info("n_gpu = {}\n".format(args.n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logging.info("Loading data...")
    start = time.time()

    if not args.converse:
        train_dataset = DatasetLoader(args, "train")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        # Shuffle=False
        train_eval_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

    test_dataset = DatasetLoader(args, "test")
    # Shuffle=False
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    logging.info("Loading data time: {:.2f}s\n".format(time.time() - start))

    logging.info("-----------------------------------------\nLoading model...\n")
    start = time.time()
    model = eval("{}(args)".format(args.model))

    criterion = nn.MSELoss(reduction="none")
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7, verbose=True
    )

    if args.n_gpu > 0:
        logging.info("Using GPU to loading model...")
        model = model.cuda()
        criterion = criterion.cuda()

    logging.info(model)
    logging.info(
        "\nLoading model time: {:.2f}s\n-----------------------------------------\n".format(
            time.time() - start
        )
    )

    if args.resume != "":
        resume_dict = torch.load(args.resume)
        model.load_state_dict(resume_dict["state_dict"])
        optimizer.load_state_dict(resume_dict["optimizer"])
        args.start_epoch = resume_dict["epoch"] + 1

    with open(os.path.join(args.save, "args.txt"), "w") as f:
        logging.info("\n---------------------------------------------------\n")
        logging.info("[Arguments] \n")
        for arg in vars(args):
            v = getattr(args, arg)
            s = "{}\t{}".format(arg, v)
            f.write("{}\n".format(s))
            logging.info(s)
        logging.info("\n---------------------------------------------------\n")
        f.write("\n{}\n".format(model))

    # Only evaluate
    if args.evaluate:
        if not args.resume:
            logging.info("No trained .pt file loaded.\n")

        logging.info("Start Evaluating..")
        args.current_epoch = resume_dict["epoch"]

        test_loss, test_y_pred, test_y_pred_file = evaluate(
            args, test_loader, model, criterion, "test"
        )
        np.save(test_y_pred_file, test_y_pred)

        if not args.converse:
            train_eval_loss, train_y_pred, train_y_pred_file = evaluate(
                args, train_eval_loader, model, criterion, "train"
            )
            np.save(train_y_pred_file, train_y_pred)

        exit()

    last_epoch = args.start_epoch if args.start_epoch != 0 else -1

    # Best results on validation dataset
    best_val_result = -np.inf
    best_val_epoch = -1

    start = time.time()
    args.global_step = 0
    for epoch in range(args.start_epoch, args.epochs):
        args.current_epoch = epoch
        logging.info("\n------------------------------------------------\n")
        logging.info(
            "Start Training Epoch {}: {}".format(
                epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            )
        )
        model.train()

        train_loss = 0.0

        lr = optimizer.param_groups[0]["lr"]
        print_step = 10
        for step, (idxs, y_gt, y_masks) in enumerate(tqdm(train_loader)):
            # y_gt: (bs, seq, output_dim)
            # y_masks: (bs, seq, 1)
            # y_pred: (bs, seq, output_dim)
            # masks: (bs, seq, 1)
            y_pred, masks = model(idxs, train_dataset)

            loss = compute_loss(criterion, y_pred, y_gt, masks * y_masks)

            if torch.any(torch.isnan(loss)):
                logging.info("out has nan: ", torch.any(torch.isnan(y_pred)))
                logging.info("y_gt has nan: ", torch.any(torch.isnan(y_gt)))
                logging.info("out: ", y_pred)
                logging.info("y_gt: ", y_gt)
                logging.info("loss = {:.4f}\n".format(loss.item()))
                exit()

            if step % print_step == 0:
                writer.add_scalar("Training/MSE Loss", loss.item(), args.global_step)

                logging.info(
                    "\n\nEpoch: {}, Step: {}, MSE loss = {:.5f}".format(
                        epoch + 1, step, loss.item()
                    )
                )
                print_idx = random.sample(range(len(y_gt)), 1)[0]
                logging.info("y_pred[{}]: {}\n".format(print_idx, y_pred[print_idx]))
                logging.info("y_gt[{}]: {}\n".format(print_idx, y_gt[print_idx]))

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            args.global_step += 1

        train_eval_loss, train_y_pred, train_y_pred_file = evaluate(
            args, train_eval_loader, model, criterion, "train"
        )
        test_loss, test_y_pred, test_y_pred_file = evaluate(
            args, test_loader, model, criterion, "test"
        )
        # Reduce learning rate when test_loss stop descreasing
        scheduler.step(test_loss)

        # Save predicting audios
        save_pred_audios_in_training(train_y_pred, args, train_eval_loss, "train")
        save_pred_audios_in_training(test_y_pred, args, test_loss, "test")

        # Save model
        val_result = -test_loss
        if val_result >= best_val_result:
            # model file
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(args.save, "{}.pt".format(epoch)),
            )
            # predictions
            np.save(test_y_pred_file, test_y_pred)
            np.save(train_y_pred_file, train_y_pred)

            if best_val_epoch != -1:
                os.system(
                    "rm {}".format(
                        os.path.join(args.save, "{}.pt".format(best_val_epoch))
                    )
                )
                os.system(
                    "rm {}".format(
                        os.path.join(
                            args.save, "{}_{}.npy".format("test", best_val_epoch)
                        )
                    )
                )
                os.system(
                    "rm {}".format(
                        os.path.join(
                            args.save, "{}_{}.npy".format("train", best_val_epoch)
                        )
                    )
                )

            best_val_result = val_result
            best_val_epoch = epoch

        # Logging
        writer.add_scalar("Training/Learning Rate", lr, epoch)
        writer.add_scalar("Evaluation/Training Dataset", train_eval_loss, epoch)
        writer.add_scalar("Evaluation/Test Dataset", test_loss, epoch)

        epoch_res = "{}\nModel: {}, Epoch: {}/{}, lr: {}".format(
            "=" * 10, args.model, epoch + 1, args.epochs, lr
        )
        epoch_res += "\n[Loss]\nTrain: {:.8f}\tTest: {:.8f}\tBest Epoch: {}\n{}".format(
            train_eval_loss, test_loss, best_val_epoch, "=" * 10
        )
        print(epoch_res)
        logging.info(epoch_res)

    logging.info("Training Time: {:.2f}s".format(time.time() - start))
    logging.info(
        "End time: {}".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        )
    )
