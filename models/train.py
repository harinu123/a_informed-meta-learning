import torch
import torch.nn.functional as F
import wandb
import numpy as np
import os
import sys
import toml
import optuna

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from dataset.utils import setup_dataloaders
from models.inp import INP
from models.loss import ELBOLoss

EVAL_ITER = 500
SAVE_ITER = 500
MAX_EVAL_IT = 50


def _randperm_no_fixed(bs, device):
    perm = torch.randperm(bs, device=device)
    fixed = perm == torch.arange(bs, device=device)
    if fixed.any():
        perm[fixed] = (perm[fixed] + 1) % bs
    return perm


def _shuffle_knowledge(knowledge, perm):
    if isinstance(knowledge, torch.Tensor):
        return knowledge[perm]
    else:
        k_list = list(knowledge)
        perm_cpu = perm.detach().cpu().tolist()
        return [k_list[i] for i in perm_cpu]


def make_fully_mismatched_knowledge(knowledge, device):
    if knowledge is None:
        return None
    bs = len(knowledge) if not isinstance(knowledge, torch.Tensor) else knowledge.shape[0]
    if bs < 2:
        return knowledge
    perm = _randperm_no_fixed(bs, device=device)
    return _shuffle_knowledge(knowledge, perm)


class Trainer:
    def __init__(self, config, save_dir, load_path=None, last_save_it=0):
        self.config = config
        self.last_save_it = last_save_it

        self.device = config.device
        self.train_dataloader, self.val_dataloader, _, extras = setup_dataloaders(
            config
        )

        for k, v in extras.items():
            config.__dict__[k] = v

        self.num_epochs = config.num_epochs

        self.model = INP(config)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        self.loss_func = ELBOLoss(beta=config.beta)
        self.loss_func_vec = ELBOLoss(reduction=None, beta=config.beta)
        if load_path is not None:
            print(f"Loading model from state dict {load_path}")
            state_dict = torch.load(load_path)
            self.model.load_state_dict(state_dict, strict=False)
            loaded_states = set(state_dict.keys())

        own_trainable_states = []
        print("Trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
                own_trainable_states.append(name)

        if load_path is not None:
            own_trainable_states = set(own_trainable_states)
            print("\n States not loaded from state dict:")
            print(
                *sorted(list(own_trainable_states.difference(loaded_states))), sep="\n"
            )
            print("Unknown states:")
            print(
                *sorted(
                    list(loaded_states.difference(set(self.model.state_dict().keys())))
                ),
                sep="\n",
            )

        self.save_dir = save_dir

    def _latent_mu(self, x_context, y_context, x_target, knowledge):
        x_context_e = self.model.x_encoder(x_context)
        x_target_e = self.model.x_encoder(x_target)
        R = self.model.encode_globally(x_context_e, y_context, x_target_e)

        q_z_stats = self.model.latent_encoder(R, knowledge, x_context.shape[1])
        mu, _rho = q_z_stats.split(self.config.hidden_dim, dim=-1)
        return mu.squeeze(1)

    def _pred_and_nll_vec(self, x_context, y_context, x_target, y_target, knowledge):
        outputs = self.model(
            x_context, y_context, x_target, y_target=y_target, knowledge=knowledge
        )
        p_yCc = outputs[0]
        pred_mean = p_yCc.mean.mean(dim=0)

        _loss_vec, _kl_vec, nll_vec = self.loss_func_vec(outputs, y_target)
        return pred_mean, nll_vec

    def get_loss(self, x_context, y_context, x_target, y_target, knowledge):
        if self.config.sort_context:
            x_context, indices = torch.sort(x_context, dim=1)
            y_context = torch.gather(y_context, 1, indices)
        if self.config.use_knowledge:
            output = self.model(
                x_context,
                y_context,
                x_target,
                y_target=y_target,
                knowledge=knowledge,
            )
        else:
            output = self.model(
                x_context, y_context, x_target, y_target=y_target, knowledge=None
            )
        loss, kl, negative_ll = self.loss_func(output, y_target)

        results = {"loss": loss, "kl": kl, "negative_ll": negative_ll}

        return results

    def run_batch_train(self, batch):
        context, target, knowledge, ids = batch
        x_context, y_context = context
        x_target, y_target = target
        x_context = x_context.to(self.device)
        y_context = y_context.to(self.device)
        x_target = x_target.to(self.device)
        y_target = y_target.to(self.device)

        if isinstance(knowledge, torch.Tensor):
            knowledge = knowledge.to(self.device)

        results = self.get_loss(x_context, y_context, x_target, y_target, knowledge)
        results["loss_total"] = results["loss"]

        use_contrastive = self.config.knowledge_contrastive and (
            self.config.kcon_inv_weight != 0.0 or self.config.kcon_use_weight != 0.0
        )
        use_functional = self.config.knowledge_functional and (
            self.config.kfunc_mismatch_weight != 0.0
            or self.config.kfunc_improve_weight != 0.0
        )

        if use_contrastive:
            k_true = knowledge
            k_null = None
            k_mis = make_fully_mismatched_knowledge(knowledge, self.device)

            mu_true = self._latent_mu(x_context, y_context, x_target, k_true)
            mu_null = self._latent_mu(x_context, y_context, x_target, k_null)
            mu_mis = self._latent_mu(x_context, y_context, x_target, k_mis)

            inv_loss = F.mse_loss(mu_mis, mu_null)
            dist = torch.norm(mu_true - mu_null, dim=-1)
            use_loss = F.relu(self.config.kcon_margin - dist).mean()

            results["kcon_inv_loss"] = inv_loss
            results["kcon_use_loss"] = use_loss
            results["loss_total"] = results["loss_total"] + (
                self.config.kcon_inv_weight * inv_loss
                + self.config.kcon_use_weight * use_loss
            )

        if use_functional:
            k_true = knowledge
            k_null = None
            k_mis = make_fully_mismatched_knowledge(knowledge, self.device)

            pred_null, nll_null = self._pred_and_nll_vec(
                x_context, y_context, x_target, y_target, k_null
            )
            pred_mis, nll_mis = self._pred_and_nll_vec(
                x_context, y_context, x_target, y_target, k_mis
            )
            pred_true, nll_true = self._pred_and_nll_vec(
                x_context, y_context, x_target, y_target, k_true
            )

            kfunc_mismatch = F.mse_loss(pred_mis, pred_null)
            kfunc_improve = F.relu(
                self.config.kfunc_margin + nll_true - nll_null
            ).mean()

            results["kfunc_mismatch_loss"] = kfunc_mismatch
            results["kfunc_improve_loss"] = kfunc_improve
            results["loss_total"] = results["loss_total"] + (
                self.config.kfunc_mismatch_weight * kfunc_mismatch
                + self.config.kfunc_improve_weight * kfunc_improve
            )

        return results

    def run_batch_eval(self, batch, num_context=5):
        context, target, knowledge, ids = batch
        x_target, y_target = target
        x_target = x_target.to(self.device)
        y_target = y_target.to(self.device)

        context_idx = np.random.choice(x_target.shape[1], num_context, replace=False)

        x_context, y_context = x_target[:, context_idx, :], y_target[:, context_idx, :]

        results = self.get_loss(x_context, y_context, x_target, y_target, knowledge)

        return results

    def train(self):
        it = 0
        min_eval_loss = np.inf
        for epoch in range(self.num_epochs + 1):
            # self.scheduler.step()
            for batch in self.train_dataloader:
                self.model.train()
                self.optimizer.zero_grad()
                results = self.run_batch_train(batch)
                loss = results.get("loss_total", results["loss"])
                kl = results["kl"]
                negative_ll = results["negative_ll"]
                loss.backward()
                self.optimizer.step()
                wandb.log({"train_loss": loss})
                wandb.log({"train_negative_ll": negative_ll})
                wandb.log({"train_kl": kl})
                if "kcon_inv_loss" in results:
                    wandb.log({"train_kcon_inv_loss": results["kcon_inv_loss"]})
                if "kcon_use_loss" in results:
                    wandb.log({"train_kcon_use_loss": results["kcon_use_loss"]})
                if "kfunc_mismatch_loss" in results:
                    wandb.log(
                        {"train_kfunc_mismatch_loss": results["kfunc_mismatch_loss"]}
                    )
                if "kfunc_improve_loss" in results:
                    wandb.log(
                        {"train_kfunc_improve_loss": results["kfunc_improve_loss"]}
                    )

                if it % EVAL_ITER == 0 and it > 0:
                    losses, val_loss = self.eval()
                    mean_eval_loss = np.mean(list(losses.values()))
                    wandb.log({"mean_eval_loss": mean_eval_loss})
                    wandb.log({"eval_loss": val_loss})
                    for k, v in losses.items():
                        wandb.log({f"eval_loss_{k}": v})

                    if val_loss < min_eval_loss and it > 1500:
                        min_eval_loss = val_loss
                        torch.save(
                            self.model.state_dict(), f"{self.save_dir}/model_best.pt"
                        )
                        torch.save(
                            self.optimizer.state_dict(),
                            f"{self.save_dir}/optim_best.pt",
                        )
                        print(f"Best model saved at iteration {self.last_save_it + it}")

                it += 1

        return min_eval_loss

    def eval(self):
        print("Evaluating")
        it = 0
        self.model.eval()
        with torch.no_grad():
            loss_num_context = [3, 5, 10]
            if self.config.min_num_context == 0:
                loss_num_context = [0] + loss_num_context
            losses_dict = dict(zip(loss_num_context, [[] for _ in loss_num_context]))

            val_losses = []
            for batch in self.val_dataloader:
                for num_context in loss_num_context:
                    results = self.run_batch_eval(batch, num_context=num_context)
                    loss = results["loss"]
                    val_results = self.run_batch_train(batch)
                    val_loss = val_results["loss"]
                    losses_dict[num_context].append(loss.to("cpu").item())
                    val_losses.append(val_loss.to("cpu").item())

                it += 1
                if it > MAX_EVAL_IT:
                    break
            losses_dict = {k: np.mean(v) for k, v in losses_dict.items()}
            val_loss = np.mean(val_losses)

        return losses_dict, val_loss


def get_device():
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda:{}".format(0))
    else:
        device = "cpu"
    print("Using device: {}".format(device))
    return device


def meta_train(trial, config, run_name_prefix="run"):
    device = get_device()
    config.device = device

    # Create save folder and save config
    save_dir = f"./saves/{config.project_name}"
    os.makedirs(save_dir, exist_ok=True)

    save_no = len(os.listdir(save_dir))
    save_no = [
        int(x.split("_")[-1])
        for x in os.listdir(save_dir)
        if x.startswith(run_name_prefix)
    ]
    if len(save_no) > 0:
        save_no = max(save_no) + 1
    else:
        save_no = 0
    save_dir = f"{save_dir}/{run_name_prefix}_{save_no}"
    os.makedirs(save_dir, exist_ok=True)

    trainer = Trainer(config=config, save_dir=save_dir)

    config = trainer.config

    # save config
    config.write_config(f"{save_dir}/config.toml")

    wandb.init(
        project=config.project_name,
        name=f"{run_name_prefix}_{save_no}",
        config=vars(config),
    )
    best_eval_loss = trainer.train()
    wandb.finish()

    return best_eval_loss


if __name__ == "__main__":
    # resume_training('run_7')
    import random
    import numpy as np
    from config import Config

    # read config from config.toml
    config = toml.load("config.toml")
    config = Config(**config)

    # set seed
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # begin study
    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda x: meta_train(x, config=config, run_name_prefix=config.run_name_prefix),
        n_trials=config.n_trials,
    )
