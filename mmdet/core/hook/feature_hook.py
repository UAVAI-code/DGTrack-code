from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class FeatureHook(Hook):

    def __init__(self, layer_list):
        self.layer_list = layer_list

    def hook_layer(self, model, selected_layer):
        def hook_function(module, grad_in, grad_out):
            if not selected_layer in model.features:
                model.features[selected_layer] = []

            model.features[selected_layer].append(grad_out)

        for n, m in model.named_modules():
            if n == str(selected_layer):
                m.register_forward_hook(hook_function)

    def hook_multi_layer(self, model, layer_list):
        for layer_name in layer_list:
            self.hook_layer(model, layer_name)

    def before_run(self, runner):
        self.hook_multi_layer(runner.model.module, self.layer_list)

