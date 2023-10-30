from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel() 
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
def calc_flops(model, input):
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    flops = FlopCountAnalysis(model, input)
    print(flops.total())
    print(flops.by_operator())
    print(flops.by_module())
    print(flops.by_module_and_operator())
    print(flop_count_table(flops))