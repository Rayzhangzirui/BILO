#!/usr/bin/env python
from datetime import datetime

# required by minlora
from minlora import *
from functools import partial

from MlflowHelper import *
from Engine import *



optobj = Options()
optobj.parse_args(*sys.argv[3:])

# save command to file
pid = os.getpid()
f = open("commands.txt", "a")
f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
f.write(f'  pid: {pid}')
f.write('\n')
f.write(' '.join(sys.argv))
f.write('\n')
f.close()


# arguments
transfer_method = sys.argv[1]
rank = int(sys.argv[2])

print(f'transfer_method: {transfer_method}, rank: {rank}')


# set up
eng = Engine(optobj.opts)
eng.setup_problem()
eng.setup_trainer()


# do not use this function, it will add lora to all layers
# add_lora(eng.net)

# this skip fflayer and fcD
if transfer_method == 'lora':
    lora_config = {
        nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear, rank=rank),
        },
    }
    eng.net.freeze_layers_except(0)
    add_lora_by_name(eng.net, ['input_layer','hidden_layers','output_layer'],lora_config = lora_config)

    eng.net.param_all = list(get_lora_params(eng.net))

elif transfer_method == 'freeze':
    # freeze all layers except the last rank-number of layer
    eng.net.freeze_layers_except(rank)

eng.trainer.config_train(eng.opts['traintype'])
summary(eng.net,verbose=2,col_names=["num_params", "trainable"])

# exit()
eng.run()


# After training, save lora 
# if use lora, save the low rank state dict, merge weight (can not be undone)
# if freeze, the state dict is already saved

if transfer_method == 'lora':
    lora_state_dict = get_lora_state_dict(eng.net)
    torch.save(lora_state_dict, get_active_artifact_path('lora_state_dict.pth'))
    merge_lora(eng.net)

    # eng.net.trainer.save()# need to merge the weight so that the prediction is correct

elif transfer_method == 'freeze':
    # net.trainer.save(artifact_dir)
    pass

else:
    raise ValueError(f'Unknown transfer transfer_method: {transfer_method}')


eng.pde.visualize(savedir=eng.logger.get_dir())