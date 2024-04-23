import torch
import copy, time
from utils import bdot

def autolirpa_opt_dp(weights, additional_coeffs, lower_bounds, upper_bounds):
    '''
    1. The upper bound has both planet and dp constraint.
    2. The lower bound has both planet constraints(function and 0).
    3. There are 2 weighting factors involved (la and ua).
    4. Input lies within a simplex.
    5. This function also optimizes the multi-neuron cut coefficients lambda.

    Optimization: this function will optimize as and ib both to get optimized lirpa bounds for this case.

    Return: this will return the upper and lower bounds
    '''
    # 1. Compute L(x, a, \lambda)
    # 2. Compute dL/da and dL/d\lambda 
    # 3. optimize over a and lambda using pgd.

    ### make the a weight tensor for all layers at once.
    ### and set their requires_grad to true.
    learning_rate_l = 1e3
    learning_rate_u = 1e3
    learning_rate_lmbd = 1e2
    lower_a = []
    upper_a = []
    lambdas = []

    for i in range(len(weights)):

        lower_a.append((upper_bounds[i] >= torch.abs(lower_bounds[i])).type(lower_bounds[i].dtype))
        lower_a[i].requires_grad = True
        upper_a.append(torch.ones_like(upper_bounds[i], requires_grad=True))## this initializes without the dp cut

        lambdas.append(torch.ones_like(upper_bounds[i].squeeze(0), requires_grad=True))

        ## empty grads at beginning
        lower_a[i].grad = None
        upper_a[i].grad = None
        lambdas[i].grad = None

    #######################
    assert len(additional_coeffs) > 0

    final_lay_idx = len(weights)
    if final_lay_idx in additional_coeffs:
        # There is a coefficient on the output of the network
        rho = additional_coeffs[final_lay_idx]
        lay_idx = final_lay_idx
    else:
        # There is none. Just identify the shape from the additional coeffs
        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]
        device = lower_bounds[-1].device

        lay_idx = final_lay_idx -1
        while lay_idx not in additional_coeffs:
            lay_shape = lower_bounds[lay_idx].shape[1:]
            lay_idx -= 1
        # We now reached the time where lay_idx has an additional coefficient
        rho = additional_coeffs[lay_idx]
    lay_idx -= 1

    initial_lay_idx = copy.deepcopy(lay_idx)
    initial_rho = copy.deepcopy(rho)

    #######################

    for it in range(50):
        # print('Iteration number: ', it)
        ##################
        ### compute L(x, a)
        ##################
        b_term = None
        rho = copy.deepcopy(initial_rho)
        lay_idx = copy.deepcopy(initial_lay_idx)
        rho_split=False

        with torch.enable_grad():
            while lay_idx > 0:  
                lay = weights[lay_idx]

                #########
                ###### for lambdas
                # stopping gradient accumulation
                lambdas[lay_idx].grad = None

                lay.update_dp_weights(lambdas[lay_idx])
                #########


                if b_term is None:
                    b_term = lay.bias_backward(rho)#rho is of size (batch_size*output_size)
                else:
                    b_term += lay.dp_bias_backward(rho_u_dp) + lay.bias_backward(rho_planet)

                if not rho_split:
                    lbda = lay.backward(rho)#rho is of size (batch_size*output_size)
                else:
                    lbda = lay.dp_backward(rho_u_dp) + lay.backward(rho_planet)
                
                lbs = lower_bounds[lay_idx]#this is of input_size as they are lower_bounds on that layer input
                ubs = upper_bounds[lay_idx]

                las = lower_a[lay_idx]
                uas = upper_a[lay_idx]

                #####
                ## beta
                beta_u = -  (uas * (lbs*ubs)) / (ubs - lbs)
                beta_u.masked_fill_(lbs > 0, 0)
                beta_u.masked_fill_(ubs <= 0, 0)

                beta_l = torch.zeros_like(lbs)

                ### POSSIBLE SPEEDUP
                #### this can be implemented as a convolution
                b_term += bdot(torch.where(lbda >= 0, beta_l.unsqueeze(1), beta_u.unsqueeze(1)), lbda)
                # if lbda.dim() == 5:
                #     b_term += torch.sum(torch.where(lbda >= 0, beta_l.unsqueeze(1), beta_u.unsqueeze(1))*lbda, dim=(-3,-2,-1))
                # else:
                #     b_term += torch.sum(torch.where(lbda >= 0, beta_l.unsqueeze(1), beta_u.unsqueeze(1))*lbda, dim=(-1))
                #####


                #####
                ## alpha
                alpha_u_planet = uas * ubs / (ubs - lbs)
                alpha_u_planet.masked_fill_(lbs > 0, 1)
                alpha_u_planet.masked_fill_(ubs <= 0, 0)

                alpha_u_dp = 1-uas
                alpha_u_dp.masked_fill_(lbs > 0, 0)
                alpha_u_dp.masked_fill_(ubs <= 0, 0)

                alpha_l = las
                with torch.no_grad():
                    alpha_l.masked_fill_(lbs > 0, 1)
                    alpha_l.masked_fill_(ubs <= 0, 0)

                ones_ten = torch.ones_like(ubs)
                zeros_ten = torch.zeros_like(ubs)

                rho_split=True
                rho_planet = torch.where(lbda >= 0, alpha_l.unsqueeze(1), alpha_u_planet.unsqueeze(1)) * lbda#(output(batch_size modulo)*input shape)
                rho_u_dp = torch.where(lbda >= 0, zeros_ten.unsqueeze(1), alpha_u_dp.unsqueeze(1)) * lbda

                # rho = torch.where(lbda >= 0, alpha_l.unsqueeze(1), alpha_u.unsqueeze(1)) * lbda#(output(batch_size modulo)*input shape)
                # rho = alpha_u.unsqueeze(1) * lbda#(output(batch_size modulo)*input shape)
                #####

                lay_idx -= 1

            ##########################
            #### compute objective
            # print(weights[1].dp_weights, b_term, rho_planet, rho_u_dp)
            bound = opt_lirpa_input_dp(weights[0], b_term, rho_planet, rho_u_dp, lower_bounds[0])
            bound_mean = bound.mean()
            # print(it, bound_mean.item())
            ##########################

            #######################################
            ###### Updating a's and lambda's ######
            #######################################
            # 1. compute gradients
            bound_mean.backward()

            # 2. update step
            with torch.no_grad():
                for i in range(1, len(lower_a)):

                    #### lower_a
                    lower_a[i] += learning_rate_l * lower_a[i].grad
                    lower_a[i] = torch.clamp(lower_a[i], 0, 1)
                    lower_a[i].requires_grad = True
                    # Manually zero the gradients after updating weights
                    lower_a[i].grad = None

                    #### upper_a
                    upper_a[i] += learning_rate_u * upper_a[i].grad
                    upper_a[i] = torch.clamp(upper_a[i], 0, 1)
                    upper_a[i].requires_grad = True
                    # Manually zero the gradients after updating weights
                    upper_a[i].grad = None

                    #######################################
                    ########## UPDATING LAMBDAS ###########
                    #######################################
                    '''
                    We don't work with lambdas of input layer and the last linear layer.
                    '''
                    if it%5==0 and i>1:
                        # update lambda
                        lambda_temp = lambdas[i-1]

                        lambda_temp += learning_rate_lmbd * lambdas[i-1].grad
                        # print(lambda_temp)
                        lambda_temp = torch.clamp(lambda_temp, 0, 1)
                        # lambda_temp = simplex_projection_sort(lambda_temp.unsqueeze(0)).squeeze(0)
                        # print(lambda_temp)
                        # input('')
                        # get alpha
                        lay_in = weights[i-2]
                        init_cut_coeff = lay_in.simplex_conditioning(lambda_temp)

                        # divide lambda by alpha
                        lambda_temp = lambda_temp/init_cut_coeff
                        # print(lambda_temp)
                        # input('')
                        lambdas[i-1] = lambda_temp

                        # Manually zero the gradients after updating weights
                        lambdas[i-1].grad = None
                        lambdas[i-1].requires_grad = True

                        # update the dp-weights
                        lay_out = weights[i-1]
                        
                        with torch.enable_grad():
                            lay_out.update_dp_weights(lambda_temp)


                    #######################################
                    #######################################

            #######################################
            #######################################

    
    return bound

def opt_lirpa_input_simplex(lay, b_term, rho):
    with torch.enable_grad():
        b_term += lay.bias_backward(rho)
        lin_eq = lay.backward(rho)

        lin_eq_matrix = lin_eq.view(lin_eq.shape[0],lin_eq.shape[1],-1)

        (b,c) = torch.min(lin_eq_matrix, 2)
        bound = b_term + torch.clamp(b, None, 0)

    return bound 

def opt_lirpa_input_dp(lay, b_term, rho_planet, rho_u_dp, inp_bound=None):
    # print("---------------------------------")
    # print(f"b_term: {b_term}")
    # print(f"rho: {rho_u_dp}")
    # print(f"input: {inp_bound}")
    with torch.enable_grad():
        b_term += lay.dp_bias_backward(rho_u_dp) + lay.bias_backward(rho_planet)

        start_time = time.time()
        lin_eq = lay.dp_backward(rho_u_dp, inp_bound)
        # print('dp backward time: ', time.time()-start_time)
        
        start_time = time.time()
        lin_eq += lay.backward(rho_planet)
        # print('backward time: ', time.time()-start_time)


        lin_eq_matrix = lin_eq.view(lin_eq.shape[0],lin_eq.shape[1],-1)

        (b,c) = torch.min(lin_eq_matrix, 2)
        bound = b_term + torch.clamp(b, None, 0)
    print("---------------------------------")
    return bound 