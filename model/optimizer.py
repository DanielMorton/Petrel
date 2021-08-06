import os
import torch
import time


class Optimizer:

    def __init__(self, model, device, config, start_epoch=0):
        self.config = config
        self.start_epoch = start_epoch

        self.base_dir = config.folder
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f"{self.base_dir}/log.csv"
        self.best_summary_loss = []

        self.model = model
        self.device = device

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr,
                                           weight_decay=4e-5)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.log(f'Fitter prepared. Device is {self.device}')

    def _start_csv(self):
        self.log("Epoch, Stage, Summary Loss, Class Loss, Box Loss\n", print_line=False)

    def _print_line(self, summary_loss, step, total_steps, stage, t):
        print(
            f"{stage} Step {step}/{total_steps}, " + \
            f"summary_loss: {summary_loss.loss.avg:.5f}, " + \
            f"class_loss: {summary_loss.class_loss.avg:.5f}, " + \
            f"box_loss: {summary_loss.box_loss.avg:.5f}, " + \
            f"time: {(time.time() - t):.5f}")

    def _log_line(self, summary_loss, epoch, stage):
        return f"{epoch}, {stage}, {summary_loss.loss.avg:.5f}," + \
               f"{summary_loss.class_loss.avg:.5f}," + \
               f"{summary_loss.box_loss.avg:.5f},"

    def fit(self, train_loader, validation_loader):
        if self.start_epoch > 0 and not self.best_summary_loss:
            self.best_summary_loss = self.validate(validation_loader)

        for epoch in range(self.start_epoch, self.config.n_epochs):
            summary_loss = self.train(train_loader)

            self.log(self._log_line(summary_loss, epoch, "Train"))
            self.save(f'{self.base_dir}/last-checkpoint.bin', epoch)

            summary_loss = self.validate(validation_loader)

            self.log(self._log_line(summary_loss, epoch, "Val"))
            if len(self.best_summary_loss) < 3:
                self.best_summary_loss.append((summary_loss.loss.avg, epoch))
                self.save(f'{self.base_dir}/best-checkpoint-{str(epoch).zfill(3)}epoch.bin', epoch)
                self.best_summary_loss.sort()
            elif summary_loss.loss.avg < self.best_summary_loss[-1][0]:
                _, old_epoch = self.best_summary_loss.pop()
                self.best_summary_loss.append((summary_loss.loss.avg, epoch))
                self.save(f'{self.base_dir}/best-checkpoint-{str(epoch).zfill(3)}epoch.bin', epoch)
                os.remove(f'{self.base_dir}/best-checkpoint-{str(old_epoch).zfill(3)}epoch.bin')
                self.best_summary_loss.sort()

            if self.config.validation_scheduler:
                self.scheduler.step()

    def validate(self, val_loader):
        self.model.eval()
        summary_loss = LossMeter()
        t = time.time()
        for step, (images, targets) in enumerate(val_loader):
            if self.config.verbose and step % self.config.verbose_step == 0:
                self._print_line(summary_loss, step, len(val_loader), "Val", t)
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                boxes = [target['bboxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]

                output = self.model(images, {'bbox': boxes, 'cls': labels,
                                             "img_scale": None,
                                             "img_size": None})
                summary_loss.update(output, batch_size)

        return summary_loss

    def train(self, train_loader):
        self.model.train()
        summary_loss = LossMeter()
        t = time.time()
        for step, (images, targets) in enumerate(train_loader):
            if self.config.verbose and step % self.config.verbose_step == 0:
                self._print_line(summary_loss, step, len(train_loader), "Train", t)

            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            boxes = [target['bboxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]

            self.optimizer.zero_grad()

            output = self.model(images, {'bbox': boxes, 'cls': labels})

            output['loss'].backward()

            summary_loss.update(output, batch_size)

            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()

        return summary_loss

    def save(self, path, epoch):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.start_epoch = checkpoint['epoch'] + 1

    def log(self, message, print_line=True):
        if self.config.verbose and print_line:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')