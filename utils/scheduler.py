import torch
import transformers


def get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
    """
    Returns the correct learning rate scheduler
    """
    scheduler = scheduler.lower()
    if scheduler == 'constantlr':
        return transformers.get_constant_schedule(optimizer)
    elif scheduler == 'warmupconstant':
        return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler == 'warmuplinear':
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=t_total)
    elif scheduler == 'warmupcosine':
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=t_total)
    elif scheduler == 'warmupcosinewithhardrestarts':
        return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                      num_training_steps=t_total)
    elif scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, mode='max')
    else:
        raise ValueError("Unknown scheduler {}".format(scheduler))