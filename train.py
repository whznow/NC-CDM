import torch
import torch.nn as nn
import torch.optim as optim
import sys
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net
import numpy as np
from sklearn.metrics import roc_auc_score
from regularizer import compute_lw_from_batch, compute_LB_from_embs
from clustering import run_kmeans_and_get_labels

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epoch_n = 20

def train():
    data_loader = TrainDataLoader()

    # ------------------- 超参数设置 -------------------
    K_student = 39
    student_reg_lambda_w = 0.0095
    student_reg_lambda_b = 0.0799
    start_student_reg_epoch = 4

    K_exer = 37
    exer_reg_lambda_w = 0.6760
    exer_reg_lambda_b = 0.1546
    start_exer_reg_epoch = 4

    # ------------------------------------------------

    net = Net(student_n, exer_n, knowledge_n).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    print('Training model...')


    loss_function = nn.NLLLoss()

    global_stu_labels = None
    global_exer_labels = None

    for epoch in range(0, epoch_n):
        data_loader.reset()
        batch_count = 0

        running_total_loss, running_loss_task = 0.0, 0.0
        running_Lw_student, running_Lb_student = 0.0, 0.0
        running_Lw_exer, running_Lb_exer = 0.0, 0.0

        # =========================
        # Step 1: classify students and exercises
        # =========================

        # --- 1.1 classify students ---
        if epoch >= start_student_reg_epoch:
            with torch.no_grad():
                embs = net.student_emb.weight.data
            global_stu_labels = run_kmeans_and_get_labels(embs, K_student, device)

        # --- 1.2 classify exercises ---
        if epoch >= start_exer_reg_epoch:
            with torch.no_grad():
                embs = net.k_difficulty.weight.data
            global_exer_labels = run_kmeans_and_get_labels(embs, K_exer, device)

        # =========================
        # Training Loop
        # =========================
        Lb_s = torch.tensor(0.0, device=device)
        Lb_exer_value = torch.tensor(0.0, device=device)

        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids = input_stu_ids.to(device)
            input_exer_ids = input_exer_ids.to(device)
            input_knowledge_embs = input_knowledge_embs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output_0 = 1 - output_1
            output = torch.cat((output_0, output_1), dim=1)
            loss_task = loss_function(torch.log(output), labels)
            total_loss = loss_task

            # ========== Student Reg ==========
            if epoch >= start_student_reg_epoch:
                # --- Lb_s (Between-Class) ---
                if batch_count % 50 == 1:
                    Lb_s = compute_LB_from_embs(net.student_emb.weight, global_stu_labels, K_student)
                total_loss = total_loss + student_reg_lambda_b * Lb_s
                running_Lb_student += Lb_s.item()

                # --- Lw_s (Within-Class) ---
                stu_embs = net.student_emb(input_stu_ids)
                batch_labels = global_stu_labels[input_stu_ids]
                Lw_s = compute_lw_from_batch(stu_embs, batch_labels)
                total_loss = total_loss + student_reg_lambda_w * Lw_s
                running_Lw_student += Lw_s.item()

            # ========== Exercise Reg ==========
            if epoch >= start_exer_reg_epoch:
                # --- Lb_e (Between-Class) ---
                if batch_count % 50 == 1:
                    Lb_exer_value = compute_LB_from_embs(net.k_difficulty.weight, global_exer_labels, K_exer)
                total_loss += exer_reg_lambda_b * Lb_exer_value
                running_Lb_exer += Lb_exer_value.item()

                # --- Lw_e (Within-Class) ---
                exer_embs = net.k_difficulty(input_exer_ids)
                batch_labels = global_exer_labels[input_exer_ids]
                Lw_e = compute_lw_from_batch(exer_embs, batch_labels)
                total_loss += exer_reg_lambda_w * Lw_e
                running_Lw_exer += Lw_e.item()

            total_loss.backward()
            optimizer.step()
            net.apply_clipper()

            if Lb_s.requires_grad:
                Lb_s = Lb_s.detach()
            Lb_exer_value = Lb_exer_value.detach()

            running_total_loss += total_loss.item()
            running_loss_task += loss_task.item()

        # =========================
        # Logging & Saving
        # =========================
        avg_total = running_total_loss / batch_count
        avg_task = running_loss_task / batch_count
        msg = f"Epoch {epoch + 1:2d} | Total: {avg_total:.4f} | Task: {avg_task:.4f}"

        if epoch >= start_student_reg_epoch:
            msg += f" | Lw_s: {running_Lw_student / batch_count:.4f}"
            msg += f" | Lb_s: {running_Lb_student / batch_count:.4f}"

        if epoch >= start_exer_reg_epoch:
            msg += f" | Lw_e: {running_Lw_exer / batch_count:.4f}"
            msg += f" | Lb_e: {running_Lb_exer / batch_count:.4f}"

        print(msg)

        rmse, auc = validate(net, epoch)
        save_snapshot(net, f'model/ASSIST/model_epoch{epoch + 1}')


def validate(model, epoch):
    data_loader = ValTestDataLoader('validation')
    net = Net(student_n, exer_n, knowledge_n)
    print('validating model...')
    data_loader.reset()
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    accuracy = correct_count / exer_count
    rmse = np.sqrt(np.mean((label_all - pred_all) **2))
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch + 1, accuracy, rmse, auc))
    with open('result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch + 1, accuracy, rmse, auc))

    return rmse, auc


def save_snapshot(model, filename):
    with open(filename, 'wb') as f:
        torch.save(model.state_dict(), f)


if __name__ == '__main__':
    if len(sys.argv) != 3 or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or not sys.argv[2].isdigit():
        print('Usage: python train.py {device} {epoch}\nExample: python train.py cuda:0 20')
        exit(1)
    else:
        device = torch.device(sys.argv[1])
        epoch_n = int(sys.argv[2])

    with open('config.txt') as f:
        f.readline()
        student_n, exer_n, knowledge_n = map(eval, f.readline().split(','))

    train()