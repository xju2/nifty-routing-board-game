import numpy as np
import torch
from torch_geometric.data import Data
import nifty_common_pb2 as pb

class NiftyConverter:
    @staticmethod
    def to_numpy(ndarray_msg: pb.NDArray):
        """Zero-copy view of the protobuf byte buffer."""
        dtype_map = {
            pb.NDArray.FLOAT32: np.float32,
            pb.NDArray.FLOAT64: np.float64,
            pb.NDArray.INT64: np.int64,
            pb.NDArray.INT32: np.int32,
            pb.NDArray.UINT8: np.uint8,
        }
        # Use frombuffer for speed; it creates a view rather than a copy
        arr = np.frombuffer(ndarray_msg.raw_data, dtype=dtype_map[ndarray_msg.dtype])
        return arr.reshape(ndarray_msg.shape)

    @classmethod
    def to_pyg(cls, proto_msg: pb.GraphBatch) -> Data:
        """Converts GraphBatch to PyTorch Geometric Data object."""
        # Extract core graph tensors
        x = torch.from_numpy(cls.to_numpy(proto_msg.nodes))
        edge_index = torch.from_numpy(cls.to_numpy(proto_msg.edge_index))

        # Optional attributes
        edge_attr = None
        if proto_msg.edge_attributes.raw_data:
            edge_attr = torch.from_numpy(cls.to_numpy(proto_msg.edge_attributes))

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Merge auxiliary data (sim results, etc.)
        for key, ndarray in proto_msg.auxiliary_data.items():
            data[key] = torch.from_numpy(cls.to_numpy(ndarray))

        return data
