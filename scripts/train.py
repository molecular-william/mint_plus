import os 
import sys
from pathlib import Path
# add project root to python path
PROJECT_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CUDA 0 = Ada (nvidia-smi shows it as 3)

from mint_plus.training.trainer import MINTTrainer

#trainer = MINTTrainer.from_config("configs/recipes/frozen_150M_opt.yaml")
#trainer = MINTTrainer.from_config("configs/recipes/frozen_150M.yaml")
#trainer = MINTTrainer.from_config("configs/recipes/frozen_35M.yaml")
#trainer = MINTTrainer.from_config("configs/recipes/no_frozen_35M.yaml")
trainer = MINTTrainer.from_config("configs/recipes/mu_fp8_8M.yaml")
#trainer = MINTTrainer.from_config("configs/recipes/frozen_lora_650M.yaml")
#trainer = MINTTrainer.from_config("configs/recipes/frozen_lora_35M.yaml")
trainer.fit()
