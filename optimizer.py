# The below code is mostly used from the Heavyball library
"""
SOAP Optimizer implementation.
"""

from itertools import chain
import torch
import torch.optim as optim

# Precondition 1d uses another 3GB (21.5GB -> 24.5GB) but doesn't change the loss curve after 1000 steps

class SOAP(optim.Optimizer):
    """
    Implements the SOAP optimizer algorithm.

    Combines Shampoo-style preconditioning with Adam-style momentum.
    """
    def __init__(self, params, lr: float = 3e-3, betas=(0.9, 0.95), shampoo_beta: float = 0.95, eps: float = 1e-8,
                 weight_decay: float = 0.01, precondition_frequency: int = 32, max_precond_dim: int = 2048,
                 merge_dims: bool = True, precondition_1d: bool = False, normalize_grads: bool = False,
                 data_format: str = "channels_first", correct_bias: bool = True, warmup_steps: int = 1):
        """
        Initializes the SOAP optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Learning rate.
            betas (tuple[float, float]): Coefficients used for computing running averages of gradient and its square (Adam part).
            shampoo_beta (float): Coefficient used for computing running averages of the preconditioner matrix (Shampoo part). If < 0, uses betas[1].
            eps (float): Term added to the denominator to improve numerical stability (Adam part).
            weight_decay (float): Weight decay (L2 penalty).
            precondition_frequency (int): Frequency (in steps) for updating the preconditioner eigenbasis (Q matrix).
            max_precond_dim (int): Maximum dimension size for preconditioning. Dimensions larger than this are skipped.
            merge_dims (bool): Whether to merge smaller adjacent dimensions for preconditioning until max_precond_dim is reached.
            precondition_1d (bool): Whether to precondition 1D tensors (like biases).
            normalize_grads (bool): Whether to normalize gradients (currently unused in the step method implementation provided).
            data_format (str): Data format ('channels_first' or 'channels_last'), relevant if merge_dims is True and tensor is 4D.
            correct_bias (bool): Whether to use bias correction for Adam moments (currently always applied based on step count).
            warmup_steps (int): Number of warmup steps for a linear learning rate scaling from 0 to base LR, applied within the step function.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= shampoo_beta < 1.0 and shampoo_beta >= 0: # Allow negative shampoo_beta to signal using beta2
             raise ValueError(f"Invalid shampoo_beta value: {shampoo_beta}")
        if not precondition_frequency >= 1:
             raise ValueError(f"Invalid precondition_frequency value: {precondition_frequency}")
        if not max_precond_dim >= 1:
             raise ValueError(f"Invalid max_precond_dim value: {max_precond_dim}")
        if not warmup_steps >= 0:
             raise ValueError(f"Invalid warmup_steps value: {warmup_steps}")
        if data_format not in ["channels_first", "channels_last"]:
             raise ValueError(f"Invalid data_format: {data_format}")


        defaults = {"lr": lr, "betas": betas, "shampoo_beta": shampoo_beta, "eps": eps, "weight_decay": weight_decay,
                    "precondition_frequency": precondition_frequency, "max_precond_dim": max_precond_dim,
                    "merge_dims": merge_dims, "precondition_1d": precondition_1d, "normalize_grads": normalize_grads,
                    "correct_bias": correct_bias, 'warmup_steps': warmup_steps}
        super().__init__(params, defaults)
        self._data_format = data_format

    def merge_dims(self, grad: torch.Tensor, max_precond_dim: int) -> torch.Tensor:
        """
        Merges dimensions of the gradient tensor until the product is less than or equal to max_precond_dim.

        Handles 'channels_first' and 'channels_last' formats for 4D tensors.

        Args:
            grad (torch.Tensor): The input gradient tensor.
            max_precond_dim (int): The maximum dimension size for merging.

        Returns:
            torch.Tensor: The gradient tensor with merged dimensions.
        """
        assert self._data_format in ["channels_first", "channels_last"]
        original_shape = grad.shape
        permuted = False
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2) # Convert to channels_first for consistent merging
            permuted = True

        current_shape = grad.shape
        new_shape = []
        curr_merged_dim = 1
        for dim_size in current_shape:
            # Try merging the current dimension
            temp_merged_dim = curr_merged_dim * dim_size
            if temp_merged_dim <= max_precond_dim:
                # If merge is valid, update the current merged dimension size
                curr_merged_dim = temp_merged_dim
            else:
                # If merging is invalid:
                # 1. If there was a previously valid merged dim, add it to new_shape
                if curr_merged_dim > 1:
                    new_shape.append(curr_merged_dim)
                # 2. Start the *new* merged dim with the current dim_size
                #    (unless the current dim itself exceeds the max)
                if dim_size <= max_precond_dim:
                     curr_merged_dim = dim_size
                else: # If current dim itself is too large, add it as is and reset merge
                     new_shape.append(dim_size)
                     curr_merged_dim = 1

        # Add the last merged dimension if it's valid
        if curr_merged_dim > 1:
            new_shape.append(curr_merged_dim)
        # Handle edge case: if the tensor had only one dimension > max_precond_dim
        elif not new_shape and current_shape:
             new_shape.append(current_shape[0])


        # Reshape the gradient - handle case where no merging occurs
        if tuple(new_shape) != current_shape:
             new_grad = grad.reshape(new_shape)
        else:
             new_grad = grad

        # Permute back if original was channels_last
        # Note: This returns the merged tensor in channels_first format if permutation happened.
        # The projection/update steps need to be consistent with this.
        # Let's return the merged grad and rely on project/project_back to handle formats.
        # if permuted:
        #     # This might be wrong if new_shape isn't 4D anymore
        #     try:
        #         new_grad = new_grad.permute(0, 2, 3, 1) # Attempt to permute back
        #     except IndexError:
        #         pass # Keep as is if not 4D

        return new_grad

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss. Optional.

        Returns:
            Optional[float]: The loss returned by the closure, if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            merged_grads_for_precond = [] # Store potentially merged grads for precond update
            projected_grads = [] # Store grads projected onto Q
            projected_exp_avgs = [] # Store exp_avgs projected onto Q

            shampoo_beta = group['shampoo_beta'] if group['shampoo_beta'] >= 0 else group["betas"][1]
            beta1, beta2 = group["betas"]
            precond_freq = group['precondition_frequency']
            max_dim = group['max_precond_dim']
            merge_dims_flag = group["merge_dims"]
            precond_1d_flag = group["precondition_1d"]

            # --- Pass 1: Initialize state, collect grads, update preconditioner matrices (GG) ---
            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad) # Keep original grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Initialize preconditioner structure (GG matrices and Q)
                    self.init_preconditioner(p.grad, state, precondition_frequency=precond_freq,
                                             shampoo_beta=shampoo_beta, max_precond_dim=max_dim,
                                             precondition_1d=precond_1d_flag, merge_dims=merge_dims_flag)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state['step'] += 1
                state_steps.append(state['step'])

                # Update the preconditioner matrices (GG) using the current gradient
                # This happens *before* projecting the gradient for the Adam update
                # Potentially clone grad if update_preconditioner modifies it inplace (it shouldn't)
                self.update_preconditioner(p.grad, state, max_precond_dim=max_dim,
                                           merge_dims=merge_dims_flag, precondition_1d=precond_1d_flag)


            # --- Pass 2: Update eigenbases (Q), project grads, Adam step ---
            for i, p in enumerate(params_with_grad):
                state = self.state[p]
                step = state['step']
                grad = grads[i] # Original gradient

                # Update eigenbasis Q periodically
                if step % precond_freq == 0:
                    state['Q'] = self.get_orthogonal_matrix_QR(state, max_precond_dim=max_dim, merge_dims=merge_dims_flag)
                    # state['Q'] = self.get_orthogonal_matrix(state['GG']) # Alternative: Use eigh


                # Project gradient onto the current eigenbasis Q for Adam update
                grad_projected = self.project(grad, state, merge_dims=merge_dims_flag, max_precond_dim=max_dim)
                projected_grads.append(grad_projected)

                # Project exp_avg onto the current eigenbasis Q (needed after Adam update)
                exp_avg_projected = self.project(exp_avgs[i], state, merge_dims=merge_dims_flag, max_precond_dim=max_dim)
                projected_exp_avgs.append(exp_avg_projected)


            # --- Adam Moment Updates (using projected gradients) ---
            # Calculate bias correction factors (using max step for safety if steps vary, though they shouldn't here)
            max_step = max(state_steps) if state_steps else 0
            bias_correction1 = 1.0 - beta1 ** max_step
            bias_correction2 = 1.0 - beta2 ** max_step

            # Update first moment estimate (exp_avg) using *original* gradients
            # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

            # Update second moment estimate (exp_avg_sq) using *projected* gradients
            # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad_projected**2)
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, projected_grads, projected_grads, value=1 - beta2)

            # Calculate Adam update step with bias correction
            # Denominator: sqrt(exp_avg_sq / bias_correction2) + eps
            denom = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_add_(denom, group["eps"]) # Add epsilon first
            if bias_correction2 > 0: # Avoid division by zero if step=0
                 torch._foreach_div_(denom, math.sqrt(bias_correction2))

            # Step size calculation with warmup and bias correction for first moment
            step_size = group["lr"] / bias_correction1 if bias_correction1 > 0 else group["lr"]
            warmup_factor = 1.0
            if group['warmup_steps'] > 0:
                warmup_factor = min(1.0, max_step / group['warmup_steps'])
            step_size *= warmup_factor
            step_size = -step_size # Use negative for update direction

            # --- Calculate Preconditioned Update Direction ---
            # Update = exp_avg_projected / denom
            # We use the projected exp_avg calculated earlier
            update_projected = torch._foreach_div(projected_exp_avgs, denom)

            # --- Project Back and Apply Update ---
            final_update = []
            for i, p in enumerate(params_with_grad):
                state = self.state[p]
                # Project the preconditioned update back to the original parameter space
                update_orig_space = self.project_back(update_projected[i], state, merge_dims=merge_dims_flag, max_precond_dim=max_dim)
                final_update.append(update_orig_space)

            # Apply the final update: p = p + step_size * final_update
            torch._foreach_add_(params_with_grad, final_update, alpha=step_size)

            # Apply weight decay (decoupled form): p = p - lr * wd * p (after main update)
            # Note: step_size already includes lr and warmup factor
            if group["weight_decay"] > 0.0:
                 wd_alpha = step_size * group["weight_decay"] / (-group["lr"] * warmup_factor) if group["lr"]*warmup_factor != 0 else 0
                 # Effective alpha for weight decay is -lr * wd
                 wd_alpha = -group["lr"] * group["weight_decay"] * warmup_factor
                 torch._foreach_add_(params_with_grad, params_with_grad, alpha=wd_alpha)


        return loss


    def init_preconditioner(self, grad: torch.Tensor, state: dict, precondition_frequency: int, shampoo_beta: float, max_precond_dim: int,
                            precondition_1d: bool, merge_dims: bool):
        """
        Initializes the preconditioner matrices (GG) and related state variables.

        Args:
            grad (torch.Tensor): A sample gradient tensor to determine shapes.
            state (dict): The optimizer state dictionary for the parameter.
            precondition_frequency (int): Frequency for updating eigenbasis Q.
            shampoo_beta (float): EMA decay for the GG matrices.
            max_precond_dim (int): Maximum dimension size to precondition.
            precondition_1d (bool): Whether to precondition 1D tensors.
            merge_dims (bool): Whether to merge dimensions before preconditioning.
        """
        state['GG'] = []  # List to hold preconditioner matrices (L and R in paper, GG here)
        processed_grad = grad # Use original grad shape info initially
        param_shape = state['exp_avg'].shape # Use parameter shape for consistency

        if grad.dim() == 1:
            # Determine shape for 1D case based on parameter shape
            dim_size = param_shape[0]
            if not precondition_1d or dim_size > max_precond_dim:
                state['GG'].append([]) # Skip 1D or large 1D tensors
            else:
                # Initialize GG for 1D
                state['GG'].append(torch.zeros(dim_size, dim_size, device=grad.device, dtype=grad.dtype))
        else:
            # Get the shape that GG matrices should correspond to (potentially merged)
            shape_for_gg = param_shape
            if merge_dims:
                 # Use merge_dims on a zero tensor with the parameter's shape to get the target merged shape
                 zero_param_like = torch.zeros(param_shape, device=grad.device, dtype=grad.dtype)
                 merged_shape_tensor = self.merge_dims(zero_param_like, max_precond_dim)
                 shape_for_gg = merged_shape_tensor.shape
                 del zero_param_like, merged_shape_tensor


            for dim_size in shape_for_gg:
                if dim_size > max_precond_dim:
                    state['GG'].append([]) # Skip large dimensions
                else:
                    # Initialize GG for this dimension
                    state['GG'].append(torch.zeros(dim_size, dim_size, device=grad.device, dtype=grad.dtype))

        state['Q'] = None  # Eigenbases will be computed later
        state['precondition_frequency'] = precondition_frequency
        state['shampoo_beta'] = shampoo_beta
        # Compute initial Q right away using identity matrices where GG is defined
        state['Q'] = self.get_orthogonal_matrix(state['GG'], identity_init=True)


    def project(self, grad: torch.Tensor, state: dict, merge_dims: bool, max_precond_dim: int) -> torch.Tensor:
        """
        Projects the gradient onto the eigenbases (Q) of the preconditioner.
        Handles merging and data format.

        Args:
            grad (torch.Tensor): The tensor (gradient or momentum) to project.
            state (dict): The optimizer state containing the eigenbases 'Q'.
            merge_dims (bool): Whether the tensor needs merging consistent with preconditioner init.
            max_precond_dim (int): Maximum dimension size used during merging.

        Returns:
            torch.Tensor: The tensor projected onto the eigenbases.
        """
        if state.get('Q') is None or not state['Q']: # Safety check if Q is missing or empty
             return grad # Cannot project

        projected_tensor = grad
        original_shape = grad.shape
        permuted_shape = None # Store shape after potential permutation for channels_last

        # --- Prepare tensor for projection (Merging and Permutation) ---
        if merge_dims:
            if grad.dim() == 4 and self._data_format == 'channels_last':
                projected_tensor = projected_tensor.permute(0, 3, 1, 2) # To channels_first
                permuted_shape = projected_tensor.shape # Store shape after permutation
            projected_tensor = self.merge_dims(projected_tensor, max_precond_dim)

        # --- Projection Loop ---
        # The loop structure assumes the tensor dimensions match the structure of state['Q']
        temp_projected_tensor = projected_tensor # Work on a temporary variable
        current_dim_index = 0 # Track which dimension of the tensor we are projecting
        for q_matrix in state['Q']:
             if len(q_matrix) > 0: # Check if this dimension was preconditioned (Q matrix exists)
                 # Project along the current dimension (index 0)
                 # tensordot contracts dims[0] of tensor with dims[1] of q_matrix
                 # Project: Q^T * tensor -> tensordot(tensor, Q, dims=[[0],[0]])
                 temp_projected_tensor = torch.tensordot(temp_projected_tensor, q_matrix, dims=[[0], [0]])
             # else: # Dimension was skipped (no Q matrix)
                  # No projection needed for this dim, just ensure the tensor is rotated
                  # so the *next* dimension to potentially project becomes dim 0.
             # Rotate tensor for the next iteration, regardless of projection
             if temp_projected_tensor.dim() > 1:
                  permute_order = list(range(1, temp_projected_tensor.dim())) + [0]
                  temp_projected_tensor = temp_projected_tensor.permute(permute_order)
             elif temp_projected_tensor.dim() == 0 : # Handle scalar tensor case
                 break # Cannot permute scalar


        projected_tensor = temp_projected_tensor # Assign final result

        # --- Reshape back if dimensions were merged ---
        # Need to reshape back to the original shape *before* merging/permutation.
        # This requires careful handling of the shape transformations.
        # The current `projected_tensor` is in the projected space, potentially rotated.
        # We need to undo the rotation and un-merge.
        # This part is complex and prone to errors. The original implementation's reshape
        # logic in project_back might be more reliable if adapted here, but let's try.

        # Reshape back to the shape *before* the projection loop's permutations.
        # If merge_dims was True, this is the merged shape.
        # If merge_dims was False, this is the original shape (potentially permuted if channels_last).

        # The rotation needs to be undone. The number of rotations depends on the tensor's final dim.
        # It's simpler to reshape directly if we know the target shape.

        # Target shape: original shape if no merging, or original shape permuted to channels_first if CL & 4D.
        target_shape_before_merge = original_shape
        if permuted_shape:
            target_shape_before_merge = permuted_shape

        if merge_dims:
            # Reshape the (potentially rotated) projected tensor back to the shape it had *before* merging.
            # Assuming projected_tensor's shape matches the merged shape structure.
            try:
                 # Attempt to reshape to the target shape before merging
                 projected_tensor = projected_tensor.reshape(target_shape_before_merge)
            except RuntimeError as e:
                 # This might fail if the projection loop rotation wasn't perfectly undone implicitly by reshape
                 # Fallback or error handling needed here. For now, log warning.
                 # print(f"Warning: Reshape after projection failed. Projected shape: {projected_tensor.shape}, Target: {target_shape_before_merge}. Error: {e}")
                 pass # Keep the tensor as is, might lead to errors later

            # If it was permuted to channels_first, permute it back now
            if permuted_shape and projected_tensor.shape == permuted_shape:
                 projected_tensor = projected_tensor.permute(0, 2, 3, 1) # To channels_last

        # If no merging, projected_tensor should already have the correct shape (potentially permuted)
        # If it was permuted, permute it back.
        elif permuted_shape and projected_tensor.shape == permuted_shape:
             projected_tensor = projected_tensor.permute(0, 2, 3, 1) # To channels_last

        # Final check: ensure output shape matches original input shape
        if projected_tensor.shape != original_shape:
            # print(f"Warning: Final projected tensor shape {projected_tensor.shape} does not match original shape {original_shape}")
            # Attempt final reshape, though it might indicate earlier issues
             try:
                 projected_tensor = projected_tensor.reshape(original_shape)
             except RuntimeError:
                 pass # Give up reshaping if dimensions don't match


        return projected_tensor

    def update_preconditioner(self, grad: torch.Tensor, state: dict, max_precond_dim: int, merge_dims: bool, precondition_1d: bool):
        """
        Updates the preconditioner matrices (GG) using an exponential moving average.

        Args:
            grad (torch.Tensor): The current gradient tensor.
            state (dict): The optimizer state containing 'GG' and 'shampoo_beta'.
            max_precond_dim (int): Maximum dimension size for preconditioning.
            merge_dims (bool): Whether dimensions were merged when initializing GG.
            precondition_1d (bool): Whether 1D tensors are preconditioned.
        """
        beta = state['shampoo_beta']
        if beta < 0: beta = self.defaults['betas'][1] # Use beta2 if shampoo_beta is negative
        lerp_factor = 1.0 - beta

        processed_grad = grad
        original_shape = grad.shape
        permuted_shape = None

        # --- Prepare Gradient for Update (Merging/Permutation to match GG structure) ---
        if merge_dims:
             if grad.dim() == 4 and self._data_format == 'channels_last':
                 processed_grad = grad.permute(0, 3, 1, 2) # To channels_first
                 permuted_shape = processed_grad.shape
             processed_grad = self.merge_dims(processed_grad, max_precond_dim)


        # --- Update GG Matrices ---
        if processed_grad.dim() == 1:
            # Handle 1D case (only if preconditioning is enabled and dim is valid)
            if precondition_1d and processed_grad.shape[0] <= max_precond_dim and len(state['GG']) > 0 and len(state['GG'][0]) > 0:
                outer_product = processed_grad.unsqueeze(1) @ processed_grad.unsqueeze(0)
                state['GG'][0].lerp_(outer_product, lerp_factor)
        else:
            # Handle multi-dimensional case
            current_grad_view = processed_grad # Tensor to perform tensordot on
            for idx, gg_matrix in enumerate(state['GG']):
                if len(gg_matrix) > 0: # Check if this dimension is preconditioned
                    # Calculate outer product for the current dimension 'idx'
                    # Contract all dimensions *except* idx
                    dims_to_contract = list(range(idx)) + list(range(idx + 1, processed_grad.dim()))

                    # Ensure tensor dimensions match the number of contraction dims
                    if len(dims_to_contract) != current_grad_view.dim() -1 :
                        # This indicates a shape mismatch, likely due to merging logic inconsistencies
                        # print(f"Warning: Dimension mismatch during GG update for dim {idx}. Skipping update.")
                        # Rotate view for next iteration even if update is skipped
                        if current_grad_view.dim() > 1:
                           permute_order = list(range(1, current_grad_view.dim())) + [0]
                           current_grad_view = current_grad_view.permute(permute_order)
                        continue


                    try:
                        outer_product = torch.tensordot(current_grad_view, current_grad_view, dims=[dims_to_contract, dims_to_contract])

                        # Ensure outer product shape matches GG matrix
                        if outer_product.shape == gg_matrix.shape:
                             gg_matrix.lerp_(outer_product, lerp_factor)
                        # else:
                        #      print(f"Warning: Shape mismatch between outer product {outer_product.shape} and GG matrix {gg_matrix.shape} for dim {idx}. Skipping update.")

                    except Exception as e:
                        # print(f"Error during tensordot for GG update (dim {idx}): {e}. Grad shape: {current_grad_view.shape}, Dims: {dims_to_contract}")
                        pass # Skip update for this dimension on error


                # Rotate the gradient view to bring the next dimension to the front for the next iteration
                # This happens regardless of whether the current dimension was updated
                if current_grad_view.dim() > 1:
                    permute_order = list(range(1, current_grad_view.dim())) + [0]
                    current_grad_view = current_grad_view.permute(permute_order)


    def project_back(self, grad: torch.Tensor, state: dict, merge_dims: bool, max_precond_dim: int) -> torch.Tensor:
        """
        Projects the gradient back from the eigenbases to the original parameter space.
        Handles merging and data format.

        Args:
            grad (torch.Tensor): The tensor (gradient or preconditioned momentum) in the eigenbasis space.
            state (dict): The optimizer state containing the eigenbases 'Q'.
            merge_dims (bool): Whether the tensor needs un-merging.
            max_precond_dim (int): Maximum dimension size used during merging.

        Returns:
            torch.Tensor: The tensor projected back to the original parameter space.
        """
        if state.get('Q') is None or not state['Q']:
            return grad # Cannot project back

        projected_back_tensor = grad
        param_shape = state['exp_avg'].shape # Target shape is the original parameter shape
        shape_before_merge = param_shape # Default if no merging/permutation
        permuted_intermediate_shape = None # Store channels-first shape if needed

        # --- Determine Shape Before Un-merging ---
        # If merging was used, the input `grad` is in the merged, projected space.
        # We need to know the shape *before* merging occurred to reshape correctly after projection.
        if merge_dims:
            if len(param_shape) == 4 and self._data_format == 'channels_last':
                # Need the channels-first shape corresponding to the parameter
                zero_param_like = torch.zeros(param_shape, device=grad.device, dtype=grad.dtype)
                permuted_intermediate_shape = zero_param_like.permute(0, 3, 1, 2).shape
                shape_before_merge = permuted_intermediate_shape
                del zero_param_like
            else:
                 shape_before_merge = param_shape


        # --- Projection Back Loop ---
        # We assume the input `grad` has the structure corresponding to the `state['Q']` matrices.
        # Iterate through Q matrices and apply projection: tensor * Q
        temp_projected_back = grad # Work on a temp variable
        for q_matrix in state['Q']:
             if len(q_matrix) > 0: # If this dimension was preconditioned
                 # Project back: tensor * Q -> tensordot(tensor, Q, dims=[[0], [1]])
                 # Contracts first dim of tensor with second dim (columns) of Q
                 try:
                     temp_projected_back = torch.tensordot(temp_projected_back, q_matrix, dims=[[0], [1]])
                 except Exception as e:
                      # print(f"Error during tensordot for project_back: {e}. Tensor shape: {temp_projected_back.shape}, Q shape: {q_matrix.shape}")
                      # If projection fails, might need to rotate and continue, or just return current state
                      # Rotate tensor for the next iteration even if projection failed
                      if temp_projected_back.dim() > 1:
                           permute_order = list(range(1, temp_projected_back.dim())) + [0]
                           temp_projected_back = temp_projected_back.permute(permute_order)
                      continue # Skip to next Q matrix

             # else: # Dimension was skipped
                  # Rotate tensor to bring next dim to front for next iteration/unmerging
             if temp_projected_back.dim() > 1:
                 permute_order = list(range(1, temp_projected_back.dim())) + [0]
                 temp_projected_back = temp_projected_back.permute(permute_order)
             elif temp_projected_back.dim() == 0:
                 break # Cannot permute scalar


        projected_back_tensor = temp_projected_back

        # --- Reshape Back to Original Parameter Shape ---
        # Current `projected_back_tensor` is potentially rotated and needs un-merging/reshaping.
        try:
             # Reshape to the shape *before* merging
             reshaped_tensor = projected_back_tensor.reshape(shape_before_merge)

             # If original was channels_last, permute back from intermediate channels-first shape
             if permuted_intermediate_shape and reshaped_tensor.shape == permuted_intermediate_shape:
                 projected_back_tensor = reshaped_tensor.permute(0, 2, 3, 1)
             else:
                 projected_back_tensor = reshaped_tensor

             # Final check: ensure shape matches original parameter shape
             if projected_back_tensor.shape != param_shape:
                  # print(f"Warning: Final project_back tensor shape {projected_back_tensor.shape} does not match parameter shape {param_shape}. Attempting final reshape.")
                  projected_back_tensor = projected_back_tensor.reshape(param_shape)

        except RuntimeError as e:
            # print(f"Error reshaping tensor in project_back. Current shape: {projected_back_tensor.shape}, Target shape before merge: {shape_before_merge}, Param shape: {param_shape}. Error: {e}")
            # Fallback: try reshaping directly to param_shape if possible
            try:
                 projected_back_tensor = projected_back_tensor.reshape(param_shape)
            except RuntimeError:
                 # print("Final reshape fallback failed.")
                 pass # Return tensor as is if all reshaping fails


        return projected_back_tensor


    def get_orthogonal_matrix(self, mat_list: list, identity_init: bool = False) -> list:
        """
        Computes the orthogonal eigenbases (Q) using torch.linalg.eigh.

        Args:
            mat_list (list): List of preconditioner matrices (GG). Can contain empty lists for skipped dims.
            identity_init (bool): If True and GG matrix is zero, initialize Q as identity.

        Returns:
            list: List of orthogonal matrices (Q), one for each preconditioned dimension.
                  Contains empty lists for skipped dimensions.
        """
        final_Q = []
        for m in mat_list:
            if not isinstance(m, torch.Tensor) or m.numel() == 0: # Handle empty lists or non-tensors
                final_Q.append([])
                continue

            # Handle non-float types if necessary, casting to float32 for decomposition
            original_type = m.dtype
            original_device = m.device
            float_m = m
            compute_dtype = torch.float32 # Default compute dtype
            if m.dtype not in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                 float_m = m.float()
            elif m.dtype in [torch.float16, torch.bfloat16]:
                 float_m = m.float() # Use float32 for computation with half precision types
            elif m.dtype == torch.float64:
                 compute_dtype = torch.float64 # Use float64 if input is float64


            # Handle identity initialization for zero matrices
            is_zero_matrix = torch.all(m == 0)
            if identity_init and is_zero_matrix:
                 Q = torch.eye(m.shape[0], device=original_device, dtype=original_type)
                 final_Q.append(Q)
                 continue

            try:
                # Add small epsilon * Identity for numerical stability during decomposition
                # Use compute_dtype for the identity matrix
                identity_eps = torch.eye(m.shape[0], device=float_m.device, dtype=compute_dtype) * 1e-30
                eig_input = float_m.to(compute_dtype) + identity_eps
                # eigh returns eigenvalues (sorted ascending), eigenvectors (columns)
                _, Q = torch.linalg.eigh(eig_input)
            except Exception as e: # Catch potential errors like non-convergence
                 # print(f"Warning: eigh failed with {compute_dtype} for tensor on {original_device} with shape {m.shape} and dtype {m.dtype}. Trying float64 fallback. Error: {e}")
                 try:
                     # Try float64 for higher precision if default failed
                     identity_eps_64 = torch.eye(m.shape[0], device=float_m.device, dtype=torch.float64) * 1e-30
                     eig_input_64 = m.double() + identity_eps_64 # Cast original m to float64
                     _, Q = torch.linalg.eigh(eig_input_64)
                 except Exception as e2:
                      # print(f"Warning: eigh failed even with float64 fallback. Returning identity. Error: {e2}")
                      Q = torch.eye(m.shape[0], device=original_device, dtype=original_type) # Fallback to identity

            # Ensure Q is on the correct device and has the original parameter dtype
            # Flip Q columns to have eigenvectors sorted by descending eigenvalues (standard practice)
            Q = torch.flip(Q, dims=[1])
            final_Q.append(Q.to(device=original_device, dtype=original_type))

        return final_Q


    def get_orthogonal_matrix_QR(self, state: dict, max_precond_dim: int, merge_dims: bool) -> list:
        """
        Computes the orthogonal eigenbases (Q) using one round of power iteration
        followed by torch.linalg.qr decomposition.

        Also reorders exp_avg_sq based on estimated eigenvalues.

        Args:
            state (dict): The optimizer state containing 'GG', 'Q', and 'exp_avg_sq'.
            max_precond_dim (int): Maximum dimension size for preconditioning.
            merge_dims (bool): Whether dimensions were merged.

        Returns:
            list: List of updated orthogonal matrices (Q).
        """
        precond_list = state['GG']
        orth_list = state.get('Q') # Previous Q

        # Fallback to eigh if previous Q doesn't exist (e.g., first update after init)
        if orth_list is None or len(orth_list) != len(precond_list):
            return self.get_orthogonal_matrix(precond_list)

        # Prepare exp_avg_sq for potential reordering
        exp_avg_sq = state['exp_avg_sq']
        param_shape = exp_avg_sq.shape # Original shape of the parameter's exp_avg_sq
        exp_avg_sq_processed = exp_avg_sq # Tensor to modify
        permuted_intermediate_shape = None

        if merge_dims:
            if exp_avg_sq.dim() == 4 and self._data_format == 'channels_last':
                exp_avg_sq_processed = exp_avg_sq.permute(0, 3, 1, 2) # To channels_first
                permuted_intermediate_shape = exp_avg_sq_processed.shape
            # Merge the potentially permuted tensor
            exp_avg_sq_processed = self.merge_dims(exp_avg_sq_processed, max_precond_dim)

        final_Q = []
        current_exp_avg_sq_view = exp_avg_sq_processed # Keep track of tensor being indexed/permuted

        # Iterate through dimensions corresponding to GG matrices
        for ind, (gg_matrix, q_matrix_prev) in enumerate(zip(precond_list, orth_list)):
            if not isinstance(gg_matrix, torch.Tensor) or gg_matrix.numel() == 0 or \
               not isinstance(q_matrix_prev, torch.Tensor) or q_matrix_prev.numel() == 0:
                # Skipped dimension
                final_Q.append([])
                # Rotate exp_avg_sq view if needed for subsequent dimensions
                if current_exp_avg_sq_view.dim() > 1:
                    permute_order = list(range(1, current_exp_avg_sq_view.dim())) + [0]
                    current_exp_avg_sq_view = current_exp_avg_sq_view.permute(permute_order)
                continue

            # Handle non-float types, compute in float32 or float64
            original_type = gg_matrix.dtype
            original_device = gg_matrix.device
            compute_dtype = torch.float32
            if original_type in [torch.float16, torch.bfloat16]:
                float_gg = gg_matrix.float()
                float_q_prev = q_matrix_prev.float()
            elif original_type == torch.float64:
                compute_dtype = torch.float64
                float_gg = gg_matrix # Already float64
                float_q_prev = q_matrix_prev # Already float64
            else: # Assume float32 or cast non-floats
                 float_gg = gg_matrix.to(torch.float32)
                 float_q_prev = q_matrix_prev.to(torch.float32)


            # Estimate eigenvalues using previous Q: diag(Q_prev^T G Q_prev)
            try:
                 est_eig = torch.diag(float_q_prev.T @ float_gg @ float_q_prev)
            except Exception as e:
                 # print(f"Warning: Failed to estimate eigenvalues for dim {ind}. Using previous Q. Error: {e}")
                 final_Q.append(q_matrix_prev) # Use previous Q as fallback
                 # Rotate exp_avg_sq view
                 if current_exp_avg_sq_view.dim() > 1:
                      permute_order = list(range(1, current_exp_avg_sq_view.dim())) + [0]
                      current_exp_avg_sq_view = current_exp_avg_sq_view.permute(permute_order)
                 continue


            sort_idx = torch.argsort(est_eig, descending=True) # Sort eigenvalues descending

            # Reorder the corresponding dimension (dim 0) of exp_avg_sq based on sorted eigenvalues
            current_exp_avg_sq_view = current_exp_avg_sq_view.index_select(0, sort_idx)

            # Reorder columns of previous Q based on sorted eigenvalues
            float_q_sorted = float_q_prev[:, sort_idx]

            # Power iteration step: G * Q_sorted
            power_iter = float_gg @ float_q_sorted

            # QR decomposition of the result G * Q_sorted = Q_new * R
            try:
                 Q_new, _ = torch.linalg.qr(power_iter)
            except Exception as e:
                 # print(f"Warning: QR decomposition failed for dim {ind}. Returning sorted previous Q. Error: {e}")
                 Q_new = float_q_sorted # Fallback to sorted previous Q


            # Ensure Q_new is on the correct device and has the original parameter dtype
            final_Q.append(Q_new.to(device=original_device, dtype=original_type))

            # Rotate the exp_avg_sq tensor view for the next dimension's processing
            if current_exp_avg_sq_view.dim() > 1:
                 permute_order = list(range(1, current_exp_avg_sq_view.dim())) + [0]
                 current_exp_avg_sq_view = current_exp_avg_sq_view.permute(permute_order)

        # --- Restore original exp_avg_sq shape and update state ---
        # current_exp_avg_sq_view now holds the reordered tensor, potentially rotated.
        try:
            # Reshape back to the shape *before* merging
            shape_before_merge = param_shape
            if permuted_intermediate_shape:
                 shape_before_merge = permuted_intermediate_shape

            reshaped_exp_avg_sq = current_exp_avg_sq_view.reshape(shape_before_merge)

            # If original was channels_last, permute back
            if permuted_intermediate_shape and reshaped_exp_avg_sq.shape == permuted_intermediate_shape:
                 final_exp_avg_sq = reshaped_exp_avg_sq.permute(0, 2, 3, 1)
            else:
                 final_exp_avg_sq = reshaped_exp_avg_sq

            # Final check and update state
            if final_exp_avg_sq.shape == param_shape:
                 state['exp_avg_sq'] = final_exp_avg_sq
            # else:
            #      print(f"Warning: Final shape of reordered exp_avg_sq {final_exp_avg_sq.shape} doesn't match param shape {param_shape}. State not updated.")


        except RuntimeError as e:
            # print(f"Error reshaping reordered exp_avg_sq: {e}. State not updated.")
            pass # Keep original exp_avg_sq state if reshape fails


        return final_Q
