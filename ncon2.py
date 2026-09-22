import numpy as np

def ncon2(tensor_list, legs_list):
    
    if len(tensor_list) != len(legs_list):
        raise ValueError("Number of tensor not matching number of legs!")
    if not tensor_list:
        raise ValueError("At least one tensor is required!")

    # Work on local lists so calling ncon2 does not change the input network.
    tensor_list = [np.asarray(tensor) for tensor in tensor_list]
    legs_list = [list(legs) for legs in legs_list]

    # perform self traces first
    for idx, current_legs in enumerate(legs_list):
        
        current_legs_array = np.asarray(current_legs)
        current_internal_leg_labels, current_label_counts = np.unique(
            current_legs_array[current_legs_array > 0], return_counts=True
        )

        if np.any(current_label_counts > 2):
            bad_legs = current_internal_leg_labels[current_label_counts > 2]
            raise ValueError(f"Tensor {idx} has leg {bad_legs} appearing more than twice!")

        legs_to_trace = current_internal_leg_labels[current_label_counts == 2]
        if len(legs_to_trace) == 0:
            continue

        for leg in legs_to_trace:
            axes_to_trace = np.flatnonzero(current_legs_array == leg)
            axis1, axis2 = axes_to_trace
            if tensor_list[idx].shape[axis1] != tensor_list[idx].shape[axis2]:
                raise ValueError(f"The two legs labelled {leg} have different dimensions!")

            tensor_list[idx] = np.trace(
                tensor_list[idx], axis1=axis1, axis2=axis2
            )

            # delete the leg label that is already traced
            current_legs_array = np.delete(current_legs_array, axes_to_trace)

        legs_list[idx] = current_legs_array.tolist()

    internal_legs = [np.array( legs_list[i] ) for i in range(len(legs_list))]
    internal_legs = np.concatenate(internal_legs)
    internal_legs = np.unique( internal_legs[internal_legs > 0] )

    while internal_legs.any():
        leg = internal_legs[0]
        tensor_locations = [
            tensor_loc
            for tensor_loc, leg_labels in enumerate(legs_list)
            if leg in leg_labels
        ]

        if len(tensor_locations) != 2:
            raise ValueError(f"Leg {leg} is appearing in tensors {tensor_locations} !")

        t1_loc, t2_loc = tensor_locations[0], tensor_locations[1]
        legs_to_contract, legs_in_t1, legs_in_t2 = np.intersect1d(
            legs_list[t1_loc], legs_list[t2_loc], return_indices=True
        )

        if np.any(legs_to_contract < 0):
            bad_legs = legs_to_contract[legs_to_contract < 0]
            raise ValueError(f"Outer leg {bad_legs} appearing in both tensor {t1_loc} and {t2_loc} !")

        tensor_list[t1_loc] = np.tensordot(
            tensor_list[t1_loc],
            tensor_list[t2_loc],
            axes=(legs_in_t1, legs_in_t2)
        )

        remaining_t1_legs = np.delete(legs_list[t1_loc], legs_in_t1)
        remaining_t2_legs = np.delete(legs_list[t2_loc], legs_in_t2)
        legs_list[t1_loc] = np.concatenate((remaining_t1_legs ,remaining_t2_legs), axis=None).tolist()

        del tensor_list[t2_loc]
        del legs_list[t2_loc]

        internal_legs = np.setdiff1d(internal_legs, legs_to_contract)

    # joining disjoint pieces
    while len(tensor_list) > 1:
        tensor_list[0] = np.tensordot(tensor_list[0], tensor_list[1], axes=0)
        legs_list[0].extend(legs_list[1])
        del tensor_list[1]
        del legs_list[1]

    if legs_list[0]:
        output_order = np.argsort(-np.asarray(legs_list[0]))
        return np.transpose(tensor_list[0], output_order)
    else: 
        return tensor_list[0].item()
