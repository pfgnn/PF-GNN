import torch 

def model_reset_parameters(model):
    def weight_reset(m):
        if hasattr(layers, 'reset_parameters'):
            layers.reset_parameters()

    for layers in model.children():
        if hasattr(layers, 'reset_parameters'):
            layers.reset_parameters()
        else:
            if isinstance(layers, torch.nn.MultiheadAttention):
                for l in layers.children():
                    l.reset_parameters()
            else:    
                for module in layers:
                    if hasattr(module, 'reset_parameters'):
                            module.reset_parameters()
                    elif isinstance(module, torch.nn.Sequential):
                        module.apply(weight_reset)
                    elif isinstance(module, torch.nn.ModuleList):
                        for mod in module:
                            if isinstance(mod, torch.nn.Sequential):
                                mod.apply(weight_reset)
                            else:            
                                mod.reset_parameters()        
                    else:
                        if not isinstance(module, torch.nn.ReLU) :#and \
                            #not isinstance(module, TargetCell):
                            print ("not reset ",module)
