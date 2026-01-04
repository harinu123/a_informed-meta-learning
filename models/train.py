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
from models.inp_attn_poe import INPAttnPoE
from models.loss import ELBOLoss

EVAL_ITER = 500
SAVE_ITER = 500
MAX_EVAL_IT = 50

MODEL_REGISTRY = {
    "inp": INP,
    "inp_attn_poe": INPAttnPoE,
}


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

        if not hasattr(config, "model_variant"):
            config.model_variant = "inp"
        if not hasattr(config, "gate_supervision"):
            config.gate_supervision = False
        if not hasattr(config, "gate_supervision_prob"):
            config.gate_supervision_prob = 0.3
        if not hasattr(config, "attn_heads"):
            config.attn_heads = 4
        if not hasattr(config, "attn_dropout"):
            config.attn_dropout = 0.0

        if (
            config.model_variant != "inp_attn_poe"
            and getattr(config, "knowledge_merge", None) == "attn_poe_gated"
        ):
            raise ValueError(
                "knowledge_merge=attn_poe_gated is only compatible with model_variant=inp_attn_poe"
            )

        model_cls = MODEL_REGISTRY.get(config.model_variant)
        if model_cls is None:
            raise ValueError(f"Unknown model variant {config.model_variant}")

        self.model = model_cls(config)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        self.loss_func = ELBOLoss(beta=config.beta)
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

    def get_loss(
        self,
        x_context,
        y_context,
        x_target,
        y_target,
        knowledge,
        apply_gate_supervision=True,
    ):
        if self.config.sort_context:
            x_context, indices = torch.sort(x_context, dim=1)
            y_context = torch.gather(y_context, 1, indices)

        gate_loss = None
        gate_target_value = None
        gate_shuffled = False
        knowledge_input = knowledge
        if (
            apply_gate_supervision
            and self.config.gate_supervision
            and getattr(self.model, "supports_gate_supervision", False)
            and knowledge is not None
        ):
            if torch.rand(1).item() < self.config.gate_supervision_prob:
                knowledge_input, gate_shuffled = self.maybe_shuffle_knowledge(knowledge)
                gate_target_value = 0.0
            else:
                gate_target_value = 1.0

        if self.config.use_knowledge:
            output = self.model(
                x_context,
                y_context,
                x_target,
                y_target=y_target,
                knowledge=knowledge_input,
            )
        else:
            output = self.model(
                x_context, y_context, x_target, y_target=y_target, knowledge=None
            )
        loss, kl, negative_ll = self.loss_func(output, y_target)

        if gate_target_value is not None:
            gate_values = self.model.get_last_gate()
            if gate_values is not None:
                gate_target = torch.full_like(gate_values, gate_target_value)
                gate_loss = F.mse_loss(gate_values, gate_target)
                loss = loss + gate_loss

        results = {"loss": loss, "kl": kl, "negative_ll": negative_ll}

        if gate_loss is not None:
            results.update(
                {
                    "gate_loss": gate_loss,
                    "tau_mean": gate_values.mean().detach(),
                    "gate_target": gate_target_value,
                    "knowledge_shuffled": gate_shuffled,
                }
            )

        return results

    def maybe_shuffle_knowledge(self, knowledge):
        if knowledge is None:
            return knowledge, False

        if torch.is_tensor(knowledge):
            perm = torch.randperm(knowledge.shape[0], device=knowledge.device)
            return knowledge[perm], True

        if isinstance(knowledge, (list, tuple)):
            perm = torch.randperm(len(knowledge)).tolist()
            shuffled = [knowledge[i] for i in perm]
            if isinstance(knowledge, tuple):
                shuffled = tuple(shuffled)
            return shuffled, True

        return knowledge, False

    def run_batch_train(self, batch):
        context, target, knowledge, ids = batch
        x_context, y_context = context
        x_target, y_target = target
        x_context = x_context.to(self.device)
        y_context = y_context.to(self.device)
        x_target = x_target.to(self.device)
        y_target = y_target.to(self.device)

        results = self.get_loss(
            x_context, y_context, x_target, y_target, knowledge, apply_gate_supervision=False
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
                loss = results["loss"]
                kl = results["kl"]
                negative_ll = results["negative_ll"]
                loss.backward()
                self.optimizer.step()
                wandb.log({"train_loss": loss})
                wandb.log({"train_negative_ll": negative_ll})
                wandb.log({"train_kl": kl})
                if "gate_loss" in results:
                    wandb.log({"train_gate_loss": results["gate_loss"]})
                    wandb.log({"train_tau_mean": results.get("tau_mean", 0)})

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
