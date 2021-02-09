import os

current = os.path.dirname(__file__)
data_path = os.path.join(current, "data")
if not os.path.exists(data_path):
    os.mkdir(data_path)