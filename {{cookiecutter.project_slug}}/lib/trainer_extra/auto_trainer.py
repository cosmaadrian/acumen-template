from .acumen_trainer import AcumenTrainer

class AutoTrainer(AcumenTrainer):

    def __init__(self, args, model):
        super().__init__(args, model)

        from lib import nomenclature

        self.losses = {
            loss_args.target_head: nomenclature.LOSSES[loss_args.kind](self.args, loss_args = loss_args.args)
            for loss_args in self.args.losses
        }

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        losses = 0

        for head_name, model_output in outputs.items():
            if head_name not in self.losses:
                continue

            label_key = 'label'
            if 'label_key' in self.losses[head_name].loss_args:
                label_key = self.losses[head_name].loss_args.label_key

            weight = 1.0
            if 'weight' in self.losses[head_name].loss_args:
                weight = self.losses[head_name].loss_args.weight

            loss = self.losses[head_name](y_true = batch[label_key], y_pred = model_output)
            loss = weight * loss

            losses = losses + loss

            self.log(f'train/{head_name}@loss', loss.item())

        final_loss = losses / len(outputs.keys())

        if len(outputs.keys()) > 1:
            self.log('train/total_loss', final_loss.item(), on_step = True)

        return final_loss
