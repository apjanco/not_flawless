from datasets import load_from_disk

def load_data():
    ds = load_from_disk('/scratch/network/aj7878/not_flawless/data/iam')
    return ds
