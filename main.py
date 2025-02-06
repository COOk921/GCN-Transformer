import os
import time
import logging
import pdb
import torch
import torch.nn as nn

from utils.Setup import setup
from utils.Optim import build_optimizer
from utils.Metric import Metrics
from dataLoader import create_dataloaders, RawGraph
from utils.config import parse_args

from model import SILN


def train(args):
    # 1. Load data
    global best_scores
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(args)
    graph = RawGraph(args).cuda()

    # 2. Build model and optimizers
    model = SILN(args)
    model = model.to(args.device)

    num_total_steps = len(train_dataloader) * args.max_epochs
    optimizer, scheduler = build_optimizer(args, model, num_total_steps)
    loss_function = nn.BCELoss()

    # 3. Training
    step = 0
    best_score = args.best_score
    start_time = time.time()

    for epoch in range(args.max_epochs):
        print('\n[ Training Epoch ', epoch, ']')
        for batch in train_dataloader:
            model.train()

            pred_user, label = model(args, batch, graph)
            loss = loss_function(pred_user, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}")

        # # 4. Validate and Test
        # v_scores = inference(args, model, val_dataloader, graph)
        # print(' # ----------Validate Result---------')
        # for metric in v_scores.keys():
        #     print(' ' + metric + ' ' + str(v_scores[metric]))

        t_scores = inference(args, model, test_dataloader, graph)
        print(' # ----------Test Result---------')
        for metric in t_scores.keys():
            print(' ' + metric + ' ' + str(t_scores[metric]))

        if sum(t_scores.values()) > best_score:
            best_score = sum(t_scores.values())
            best_scores = t_scores
            torch.save({'model_state_dict': model.state_dict()}, f'{args.saved_model_path}/{args.dataset}.bin')
            print(' --> Save Model <-- ')

    print('\n #-------Reported Result-------')
    for metric in best_scores.keys():
        print(' ' + metric + ' ' + str(best_scores[metric]))


def inference(args, model, dataloader, graph):
    model.eval()
    k_list = args.metric_k
    scores = {}
    for k in k_list:
        scores['hit@' + str(k)] = 0
        scores['map@' + str(k)] = 0
    n_total_words = 0
    with torch.no_grad():
        for batch in dataloader:
            prediction, label = model(args, batch, graph)
            scores_batch, scores_len = Metrics(args).compute_metric(prediction.cpu(),
                                                                    label.contiguous().detach().cpu().numpy())
            n_total_words += scores_len
            for k in k_list:
                scores['hit@' + str(k)] += scores_batch['hit@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len
    for k in k_list:
        scores['hit@' + str(k)] = scores['hit@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    return scores


def main():
    args = parse_args()
    setup(args)

    os.makedirs(args.saved_model_path, exist_ok=True)
    logging.info("Training/Testing parameters: %s", args)
    train(args)


if __name__ == '__main__':
    main()
