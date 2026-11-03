from .configuration_llama3 import Llama3Config
from .modeling_llama3 import Llama3ForCausalLM, Llama3Model

# On save_pretrained this writes auto_map into config.json and copies the code files next
# to the weights, so AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
# works from a local folder or from the Hub.
Llama3Config.register_for_auto_class()
Llama3ForCausalLM.register_for_auto_class('AutoModelForCausalLM')

__all__ = ['Llama3Config', 'Llama3ForCausalLM', 'Llama3Model']
