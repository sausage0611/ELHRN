import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn import NLLLoss


def ti_train(train_loader, model,
                  optimizer, writer, iter_counter):
    way = model.way
    query_shot = model.shots[-1]
    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()
    criterion = NLLLoss().cuda()

    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('lr', lr, iter_counter)
    writer.add_scalar('W1', model.w1.item(), iter_counter)
    writer.add_scalar('W2', model.w2.item(), iter_counter)
    writer.add_scalar('W3', model.w3.item(), iter_counter)
    writer.add_scalar('W4', model.w4.item(), iter_counter)
    writer.add_scalar('scale', model.scale.item(), iter_counter)

    avg_loss = 0
    avg_acc = 0

    for i, (inp, _) in enumerate(train_loader):
        iter_counter += 1

        inp = inp.cuda()
        log_prediction = model(inp)

        loss = criterion(log_prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()

        _, max_index = torch.max(log_prediction, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way

        avg_acc += acc
        avg_loss += loss_value

    avg_acc = avg_acc / (i + 1)
    avg_loss = avg_loss / (i + 1)
    print('avg_acc ', avg_acc)
    print('avg_loss ', avg_loss)
    writer.add_scalar('proto_loss', avg_loss, iter_counter)
    writer.add_scalar('train_acc', avg_acc, iter_counter)

    return iter_counter, avg_acc


def auto_train(train_loader, model,optimizer, writer, iter_counter):

    criterion = NLLLoss().cuda()

    writer.add_scalar('scale', model.scale.item(), iter_counter)
    avg_loss = 0
    avg_acc = 0


    combinations = [(5, 1), (5, 5), (5, 10), (5, 15), (5, 20), (5, 25), (5, 30),
                    (10, 5), (15, 5), (20, 5), (25, 5), (30, 5)]
    #
    acc_dict = {comb: [] for comb in combinations}
    loss_dict = {comb: [] for comb in combinations}
    avg_acc_dict = {scenario: 0 for scenario in combinations}
    avg_loss_dict = {scenario: 0 for scenario in combinations}


    for i, (inp, labels) in enumerate(train_loader):

        iter_counter += 1

        inp = inp.cuda()


        #5way 1shot  inp.size(0)==5*(1+15)=80
        if inp.size(0) == 80:
            way = 5
            shot = 1
        #5way 5shot  5*(5+15)=100
        elif inp.size(0) == 100:
            way = 5
            shot = 5
        #5way 10shot  5*(10+15)=125
        elif inp.size(0) == 125:
            way = 5
            shot = 10
        #5way 15shot   5*(15+15)=150
        elif inp.size(0) == 150:
            way = 5
            shot = 15
        #5way 20shot 5*(20+15)=175
        elif inp.size(0) == 175:
            way = 5
            shot = 20
        #5way 25shot 5*(25+15)=200
        elif inp.size(0) == 200 and labels[0] == labels[125]:
            way = 5
            shot = 25
        #5way 30shot 5*(30+15)=225
        elif inp.size(0) == 225:
            way = 5
            shot = 30
        #10way 5shot 10*(5+15)=200
        elif inp.size(0) == 200 and labels[0] == labels[50]:
            way = 10
            shot = 5
        #15way 5shot 15*(5+15)=300
        elif inp.size(0) == 300:
            way = 15
            shot = 5
        #20way 5shot 20*(5+15)=400
        elif inp.size(0) == 400:
            way = 20
            shot = 5
        #25way 5shot 25*(5+15)=500
        elif inp.size(0) == 500:
            way = 25
            shot = 5
        #30way 5shot 30*(5+15)=600
        elif inp.size(0) == 600:
            way = 30
            shot = 5


        query_shot = 15
        trial = 0
        target = torch.LongTensor([j // query_shot for j in range(query_shot * way)]).cuda()

        log_prediction = model(inp, way, shot, query_shot)

        loss = criterion(log_prediction, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        loss_value = loss.item()

        _, max_index = torch.max(log_prediction, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way


        acc_dict[(way, shot)].append(acc)
        loss_dict[(way, shot)].append(loss_value)
    for (way, shot), acc_list in acc_dict.items():
        if not acc_list:
            avg_acc = 0
        else:
            avg_acc = sum(acc_list) / len(acc_list)
        avg_acc_dict[(way, shot)] = avg_acc

        if not loss_dict[(way, shot)]:
            avg_loss = 0
        else:
            avg_loss = sum(loss_dict[(way, shot)]) / len(loss_dict[(way, shot)])
        avg_loss_dict[(way, shot)] = avg_loss

    writer.add_scalar('5way1shot', avg_loss_dict[(5, 1)], iter_counter)
    writer.add_scalar('5way5shot', avg_loss_dict[(5, 5)], iter_counter)
    writer.add_scalar('5way10shot', avg_loss_dict[(5, 10)], iter_counter)
    writer.add_scalar('5way15shot', avg_loss_dict[(5, 15)], iter_counter)
    writer.add_scalar('5way20shot', avg_loss_dict[(5, 20)], iter_counter)
    writer.add_scalar('5way25shot', avg_loss_dict[(5, 25)], iter_counter)
    writer.add_scalar('5way30shot', avg_loss_dict[(5, 30)], iter_counter)
    writer.add_scalar('10way5shot', avg_loss_dict[(10, 5)], iter_counter)
    writer.add_scalar('15way5shot', avg_loss_dict[(15, 5)], iter_counter)
    writer.add_scalar('20way5shot', avg_loss_dict[(20, 5)], iter_counter)
    writer.add_scalar('25way5shot', avg_loss_dict[(25, 5)], iter_counter)
    writer.add_scalar('30way5shot', avg_loss_dict[(30, 5)], iter_counter)

    return iter_counter, avg_acc_dict, avg_loss_dict
