import torch
import math

def bdot(elt1, elt2):
    # Batch dot product
    return (elt1 * elt2).view(*elt1.shape[:2], -1).sum(-1)

def create_final_coeffs_slice(start_batch_index, end_batch_index, batch_size, nb_out, tensor_example, node_layer_shape, node=(-1, None), upper_bound=False):
    # Given indices and specifications (batch size for BaB, number of output neurons, example of used tensor and shape
    # of current last layer), create a slice of final_coeffs for dual iterative solvers (auxiliary fake last layer
    # variables indicating which neurons to compute bounds for)
    is_batch = (node[1] is None)
    if not is_batch:
        slice_coeffs = torch.zeros_like(tensor_example).unsqueeze(1)
        if tensor_example.dim() == 2:
            slice_coeffs[:, 0, node[1]] = -1 if upper_bound else 1
        elif tensor_example.dim() == 4:
            slice_coeffs[:, 0, node[1][0], node[1][1], node[1][2]] = -1 if upper_bound else 1
        else:
            raise NotImplementedError
    else:
        slice_indices = list(range(start_batch_index, end_batch_index)) if is_batch else 0 # TODO
        slice_coeffs = torch.zeros((len(slice_indices), nb_out),
                                   device=tensor_example.device, dtype=tensor_example.dtype)
        slice_diag = slice_coeffs.diagonal(start_batch_index)#returns the diagonal
        slice_diag[:] = -torch.ones_like(slice_diag)
        slice_diag = slice_coeffs.diagonal(start_batch_index - nb_out)
        slice_diag[:] = torch.ones_like(slice_diag)
        slice_coeffs = slice_coeffs.expand((batch_size, *slice_coeffs.shape))
        slice_coeffs = slice_coeffs.view((batch_size, slice_coeffs.size(1),) + node_layer_shape)
        # # let us say that outshape is 4 and batch_size is 6, then these are the 2 matrices this function will produce.
        # # after changing the shape to output shape
        # it shows that in the first output we will keep lower bound of first neuron. then lower bound of second and so on.
        # -1 0  0  0
        # 0  -1 0  0
        # 0  0  -1 0
        # 0  0  0  -1
        # 1  0  0  0
        # 0  1  0  0
        # 0  0  1  0
        # 0  0  0  1
    return slice_coeffs