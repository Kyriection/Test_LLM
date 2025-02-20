# galore optimizer
from .adafactor import Adafactor as GaLoreAdafactor
from .adamw import AdamW as GaLoreAdamW
from .adamw8bit import AdamW8bit as GaLoreAdamW8bit

# q-galore optimizer
from .q_galore_adamw8bit import AdamW8bit as QGaLoreAdamW8bit
from .simulate_q_galore_adamw8bit import AdamW8bit as QGaLoreAdamW8bit_simulate

from .SPAM import AdamW as SPAMAdam
from .stablespam import AdamW as STABLEAdam
from .stablespam1 import AdamW as STABLEAdam1
from .adam_mini_haotian import Adam_mini as Adam_mini_our