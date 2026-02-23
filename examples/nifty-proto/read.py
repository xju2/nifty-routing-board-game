import nifty_common_pb2 as pb
from utils import NiftyConverter

with open("nifty_common.bin", "rb") as f:
    batch = pb.GraphBatch()
    batch.ParseFromString(f.read())

    converter = NiftyConverter()
    data = converter.to_pyg(batch)
    print("GraphBatch converted to PyTorch Geometric Data object:", data)

# print(batch)
