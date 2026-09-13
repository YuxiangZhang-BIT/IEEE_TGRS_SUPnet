import os
import numpy as np
from tqdm import tqdm
import torch
from utils.utils_HSI import metrics
import scipy.io as io
from utils.utils_HSI import grouper, sliding_window, count_sliding_window
from models.generator import SUGenerator
import time
def train(network, num_classes, train_loader, test_loader, opts, saving_path, device, writer=None):

    best_test_acc = 0
    losses = []
    list_SUG_loss = []
    list_MPL_losses = []
    SUG = SUGenerator(opts.num_em, opts.num_bands, opts.z_dim, opts.num_domains).to(device)
    SUG.apply(SUG.weights_init)
    SUG_opt = torch.optim.Adam(SUG.parameters(), opts.lr)

    n_parameters_wo=sum(p.numel() for name, p in network.named_parameters() if p.requires_grad)
    print('MPL number of params (M): %.2f' % ((n_parameters_wo) / 1.e6))
    n_parameters_wo=sum(p.numel() for name, p in SUG.named_parameters() if p.requires_grad)
    print('SUG number of params (M): %.2f' % ((n_parameters_wo) / 1.e6))

    for e in tqdm(range(1, opts.epoch+1), desc="training the network"):

        network.train()
        SUG.train()
        test_start = time.time()
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            img = images.reshape(-1,opts.num_bands)
            img_hat, mu, log_var, Abu, em_tensor = SUG(img, device)
            if 'NW' in opts.source_name:
                img_hat_ = (img_hat - torch.min(img_hat)) / (torch.max(img_hat) - torch.min(img_hat))
                img_ = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
                loss_rec = ((img_hat_ - img_) ** 2).sum() / img_.shape[0]
            else:
                loss_rec = ((img_hat - img) ** 2).sum() / img.shape[0]

            kl_div = -0.5 * (log_var + 1 - mu ** 2 - log_var.exp())
            kl_div = kl_div.sum() / img.shape[0]
            kl_div = torch.max(kl_div, torch.tensor(0.2).to(device))
            loss_SUG = opts.lambda_1 * (loss_rec + kl_div)

            if e < opts.epoch_pgen:
                SUG_opt.zero_grad()
                loss_SUG.backward()
                SUG_opt.step()
                list_SUG_loss.append(loss_SUG.item())
            else:
                em_tensor, img_gen_all, EM_interp_all = SUG(img, device, gen=True)
                img_all = [img_gen_all[i].reshape(images.shape[0],1,opts.num_bands,opts.patch_size,opts.patch_size) for i in range(len(img_gen_all))]
                EM_interp_all.append(em_tensor)
                EM_all = [EM_interp_all[i].reshape(images.shape[0],1,opts.num_em,opts.num_bands,opts.patch_size,opts.patch_size) for i in range(len(img_gen_all)+1)]
                img_all.append(images)
                img_all = torch.concatenate(img_all,0)
                EM_all = torch.concatenate(EM_all,0).reshape(-1,opts.num_bands,opts.patch_size,opts.patch_size)
                targets_all = targets.repeat(opts.num_domains+1)
                em_targets_all = torch.cat([targets_all[i].repeat(opts.num_em) for i in range(targets_all.shape[0])])
                domain_labels = torch.cat([torch.full((images.shape[0], ), i, dtype=torch.int64, device=int(opts.device)) for i in range(opts.num_domains+1)])
                em_labels = torch.cat([torch.full((images.shape[0]*opts.num_em, ), i, dtype=torch.int64, device=int(opts.device)) for i in range(opts.num_domains+1)])
                SUG_opt.zero_grad()
                loss_cls, loss_prompt, loss_weight = network.update(img_all, targets_all, EM_all, domain_labels, em_labels, opts, em_targets_all)
                loss_MPL = loss_prompt+loss_cls+loss_weight
                loss_SUG.backward()
                SUG_opt.step()
                list_SUG_loss.append(loss_SUG.item())
                list_MPL_losses.append(loss_MPL)
                losses.append((loss_SUG+loss_MPL).item())

        test_end = time.time()
        print("training time per DataSet(s): " + "{:.5f}".format(test_end-test_start))

        if e > opts.epoch_pgen:
            mean_losses = np.mean(losses)
            mean_SUG_loss = np.mean(list_SUG_loss)
            mean_MPL_loss = np.mean(list_MPL_losses)
            train_info = "train at epoch {}/{}, loss={:.6f}, SUG_loss={:.6f}, MPL_loss={:.6f}"
            train_info = train_info.format(
                e, opts.epoch,  mean_losses, mean_SUG_loss, mean_MPL_loss)
            train_MPL_info = "train at epoch {}/{}, MPL loss={:.6f}, loss_prompt={:.6f}, loss_cls={:.6f}, loss_weight={:.6f},"
            train_MPL_info = train_MPL_info.format(
                e, opts.epoch,  mean_MPL_loss, loss_prompt, loss_cls, loss_weight)
            tqdm.write(train_info)
            tqdm.write(train_MPL_info)

            writer.add_scalar('total loss', mean_losses, e)
            writer.add_scalar('mean_SUG_loss', mean_SUG_loss, e)
            writer.add_scalar('mean_MPL_loss', mean_MPL_loss, e)
            writer.add_scalar('loss_weight', loss_weight, e)

            losses = []
        else:
            losses = []
            mean_SUG_loss = np.mean(list_SUG_loss)
            train_info = "train at epoch {}/{}, SUG_loss={:.6f}"
            train_info = train_info.format(
                e, opts.epoch, mean_SUG_loss)
            tqdm.write(train_info)
            mean_losses, mean_MPL_loss, mean_SUG_loss, loss_w, loss_w_em = 0, 0, 0, 0, 0

        if e > opts.epoch_pgen and e % opts.log_interval == 0:
            ts_acc, results = validation(network, SUG, test_loader, device, num_classes, opts, show=True)
            is_best = ts_acc >= best_test_acc
            print(f'best test OA = {best_test_acc:.4f}')
            if ts_acc > best_test_acc:
                best_test_acc = max(ts_acc, best_test_acc)
                save_ts_checkpoint(network, SUG, is_best, saving_path,
                                epoch=e, acc=best_test_acc, tmp_acc=ts_acc, seed=opts.seed)

                io.savemat(os.path.join(saving_path,
                                        'results_' + f'_{best_test_acc:.4f}_{opts.seed}' +'.mat'),
                            {'lr': opts.lr, 'lambda_1': opts.lambda_1, 'lambda_2': opts.lambda_2, 'depth': opts.depth, 're_ratio':opts.re_ratio, 'results': results, 'seed': opts.seed, 'num_tokens': opts.num_tokens,})
                writer.add_scalar('best_test_acc', best_test_acc, e)
            writer.add_scalar('test_acc', ts_acc, e)


    writer.close()
    return best_test_acc

def validation(network, SUG, val_loader, device, num_classes, opts, show=False):
    num_correct = 0.
    total_num = 0.
    network.eval()
    SUG.eval()
    ps = []
    ys = []
    for batch_idx, (images, targets) in enumerate(val_loader):
        images, targets = images.to(device), targets.to(device)
        img = images.reshape(-1,opts.num_bands)
        _, _, _, _, em_tensor = SUG(img, device)
        EM = em_tensor.reshape(-1,opts.num_bands,opts.patch_size,opts.patch_size)
        outputs = network.predict(images, EM)
        _, outputs = torch.max(outputs, dim=1)
        ps.append(outputs.detach().cpu().numpy())
        ys.append(targets.detach().cpu().numpy())
        for output, target in zip(outputs, targets):
            num_correct = num_correct + (output.item() == target.item())
            total_num = total_num + 1
    overall_acc = num_correct / total_num
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    if show:
        results = metrics(ps, ys, n_classes=num_classes)
        # show_results(run_results, label_values=labels)
        print(results['Confusion_matrix'], '\n', 'TPR:', np.round(
            results['TPR']*100, 2), '\n', 'OA:', results['Accuracy'])
        return overall_acc, results
    return overall_acc


def test(network, model_dir, image, patch_size, n_classes, device):
    network.load_state_dict(torch.load(model_dir + "/model_best.pth"))
    network.eval()

    patch_size = patch_size
    batch_size = 64
    window_size = (patch_size, patch_size)
    image_w, image_h = image.shape[:2]
    pad_size = patch_size // 2

    # pad the image
    image = np.pad(image, ((pad_size, pad_size),
                   (pad_size, pad_size), (0, 0)), mode='reflect')

    probs = np.zeros(image.shape[:2] + (n_classes, ))

    iterations = count_sliding_window(
        image, window_size=window_size) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(image, window_size=window_size)),
                      total=iterations,
                      desc="inference on the HSI"):
        with torch.no_grad():
            data = [b[0] for b in batch]
            data = np.copy(data)
            data = data.transpose((0, 3, 1, 2))
            data = torch.from_numpy(data)
            data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = network(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu').numpy()

            for (x, y, w, h), out in zip(indices, output):
                probs[x + w // 2, y + h // 2] += out
    return probs[pad_size:image_w + pad_size, pad_size:image_h + pad_size, :]


def save_checkpoint(network, is_best, saving_path, **kwargs):
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path, exist_ok=True)

    if is_best:
        tqdm.write(
            "epoch = {epoch}: best validation OA = {acc:.4f}".format(**kwargs))
        torch.save(network.state_dict(), os.path.join(
            saving_path, 'model_best.pth'))
    else:  # save the ckpt for each 10 epoch
        if kwargs['epoch'] % 10 == 0:
            torch.save(network.state_dict(), os.path.join(
                saving_path, 'model.pth'))


def save_ts_checkpoint(network, SUG, is_best, saving_path, **kwargs):
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path, exist_ok=True)

    if is_best:
        tqdm.write(
            "epoch = {epoch}: best test OA = {acc:.4f}".format(**kwargs))
        torch.save({'MPL':network.state_dict(), 'SUG':SUG.state_dict()}, os.path.join(
            saving_path, 'model_ts_best{acc:.4f}_{seed}.pth'.format(**kwargs)))
    else:  # save the ckpt for each 10 epoch
        tqdm.write(
            "epoch = {epoch}: best test OA = {acc:.4f}".format(**kwargs))
        tqdm.write(
            "epoch = {epoch}: temp test OA = {tmp_acc:.4f}".format(**kwargs))

