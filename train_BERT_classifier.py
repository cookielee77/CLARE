import os
import time
import pickle

from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup

from config import load_arguments
from utils.utils import AverageMeter
from utils.hyper_parameters import nclasses
from dataloaders.BERT_cls_loader import BERTClsDataloader

torch.manual_seed(2020)
np.random.seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def evaluation(args, model, dataloader):
    model.eval()
    pred_index = []
    pred_probs = []
    with torch.no_grad():
        losses = AverageMeter()
        acc = AverageMeter()
        begin_time = time.time()
        for i, data in enumerate(dataloader):
            input_ids = data[0].cuda()
            input_mask = data[1].cuda()
            token_ids = data[2].cuda()
            labels = data[3].cuda()
            
            loss, logits = model(input_ids=input_ids,
                                 attention_mask=input_mask,
                                 token_type_ids=token_ids,
                                 labels=labels)

            # caculate probs
            batch_preds = F.softmax(logits, dim=-1, dtype=torch.float32).cpu().detach()
            batch_probs, batch_index = torch.max(batch_preds, dim=-1)
            # calcualte accuracy
            correct_counts = torch.sum(batch_index == labels.cpu().detach()).numpy()
            acc.update(correct_counts, len(labels))
            losses.update(loss.item(), 1)
            pred_index.append(batch_index.numpy())
            pred_probs.append(batch_probs.numpy())
    print('Loss %.5f\t Acc %.2f\t Time %.3f' % (losses.avg, acc.avg * 100, time.time() - begin_time))
    return losses.avg, acc.avg, np.concatenate(pred_index), np.concatenate(pred_probs)


if __name__ == '__main__':
    args = load_arguments()
    if not os.path.exists(args.target_model_path):
        os.makedirs(args.target_model_path)
    
    # dataset
    cache_path = os.path.join('./tmp', args.dataset + '_dataset.cache.pkl')
    if os.path.exists(cache_path) and args.load_dataset_from_cache:
        with open(cache_path, 'rb') as f:
            dataloaders = pickle.load(f)
    else:
        dataset = BERTClsDataloader(args.case)
        dataloaders = dataset.get_training_dataloaders(args)
        # with open(cache_path, 'wb') as f:
        #     pickle.dump(dataloaders, f, pickle.HIGHEST_PROTOCOL)
    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['valid']
    test_dataloader = dataloaders['test']

    # model and optimization
    if 'bert' in args.target_model:
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-'+args.case, num_labels=nclasses[args.dataset],
            output_attentions=False, output_hidden_states=False).cuda()
    elif args.target_model == 'textcnn':
        from models.textcnn import TextCNN
        model = TextCNN(len(dataset.tokenizer), nclasses[args.dataset]).cuda()

    optimizer = AdamW(model.parameters(), lr=args.cls_lr)
    if 'bert' in args.target_model:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * args.cls_epochs)

    # train and evaluation output frequence
    cls_train_freq = int(len(train_dataloader) / 10)
    cls_eval_freq = int(len(train_dataloader) / 3)

    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    best_acc = 0
    for epoch in range(args.cls_epochs):
        print('------------------------- Epoch %d ------------------------- ' % epoch)
        batch_time.reset()
        losses.reset()
        end_time = time.time()
        for i, data in enumerate(tqdm(train_dataloader)):
            input_ids = data[0].cuda()
            input_mask = data[1].cuda()
            token_ids = data[2].cuda()
            labels = data[3].cuda()
            # train step
            model.zero_grad()            
            loss, logits = model(input_ids=input_ids,
                                 attention_mask=input_mask,
                                 token_type_ids=token_ids,
                                 labels=labels)
            loss.backward()
            optimizer.step()
            if 'bert' in args.target_model:
                scheduler.step()

            # udpate meters
            losses.update(loss.item(), 1)
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            
            # logging output
            if i % cls_train_freq == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                        epoch, i, len(train_dataloader), batch_time=batch_time,
                        speed=len(labels)/batch_time.val,
                        loss=losses)
                print(msg)

            # if epoch > 0 and i % cls_eval_freq == 0 and i != 0:
        print('Valiation Step %d: ' % i)
        _, acc, _, _ = evaluation(args, model, val_dataloader)
        # save model
        if acc > best_acc and args.save_model:
            best_acc = acc
            print('Saving Model ...')
            model.save_pretrained(args.target_model_path)
        model.train()

    # use the best model testing
    print("Loading best model from %s" % args.target_model_path)
    if 'bert' in args.target_model:
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(
            args.target_model_path, num_labels=nclasses[args.dataset],
            output_attentions=False, output_hidden_states=False).cuda()
    elif args.target_model == 'textcnn':
        from models.textcnn import TextCNN
        model = TextCNN(len(dataset.tokenizer), nclasses[args.dataset])
        model.load_state_dict(torch.load(
            os.path.join(args.target_model_path, 'model.pt')))
        model = model.cuda()
    print("Evaluating the best model on validation set ...")
    evaluation(args, model, val_dataloader)
    print("Evaluating the best model on test set ...")
    evaluation(args, model, test_dataloader)
    if not args.save_model:
        os.system('rm -rf %s' % args.target_model_path)