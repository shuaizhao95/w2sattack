
import logging
import os
def model_log_ca(name, eval, epoch, accuracy):
    log_dir = "logs"
    filename = f"{name}_log.log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, filename)
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info('CA: %s, epoch: %s, accuracy: %.4f', eval, epoch, accuracy)

def model_log_asr(name, eval, epoch, accuracy):
    log_dir = "logs"
    filename = f"{name}_log.log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, filename)
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info('ASR: %s, epoch: %s, attack: %.4f', eval, epoch, accuracy)


def parameter(name, ce, dl, cl):
    log_dir = "logs"
    filename = f"{name}_parameter.log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, filename)
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info('ce_loss: %s, distillation_loss: %s, con_loss: %s', ce, dl, cl)
    # loss = 5.0*ce_loss + 5.0*distillation_loss + 0.005*con_loss