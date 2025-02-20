""" AdamW Optimizer
Impl copied from PyTorch master

NOTE: Builtin optim.AdamW is used by the factory, this impl only serves as a Python based reference, will be removed
someday
"""
import math
import torch
from torch.optim.optimizer import Optimizer
import torch.optim as optim
class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.5, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max+1, eta_min, last_epoch)
        self.T_max=T_max
        self.eta_min=eta_min
    def step(self,current_step):
        self.cosine_stepper.step(current_step)

    def get_dr(self,current_step):
        self.step(current_step)
        return self.sgd.param_groups[0]['lr']

class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False,gamma1=0.9,gamma2=0.999,theta=0.999,total_T=20000,eta_min=0.5,update_proj_gap=1000):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)
        self.gamma1=gamma1 # 0.85 & 0.5 & 0.8,0.9
        self.gamma2=gamma2 # 0.99999 # 0.999,0.9999
        self.theta=theta # 0.999
        self.warmup=CosineDecay(1.0,total_T,eta_min=eta_min)   #total_T is the totoal number of update steps
        self.total_steps=0
        if self.gamma1==-1:
            self.gamma1=betas[0]
        # self.init_masks()
        self.update_proj_gap=update_proj_gap
        print("hyperparameters:",gamma1,gamma2,theta,update_proj_gap)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    def init_masks(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "rank" in group:
                    if 'mask' not in state:
                        assert len(p.data.shape)==2
                        state['mask']=self.initialize_random_rank_boolean_tensor(p.data.shape[0],p.data.shape[1],group['rank']).to(p.device)
        
    def initialize_random_rank_boolean_tensor(self, m, n, density):
        total_elements = m * n
        non_zero_count = int(density * total_elements)
        
        tensor = torch.zeros((m, n), dtype=torch.bool)
        
        non_zero_count = min(non_zero_count, total_elements)
        
        if non_zero_count > 0:
            # Generate unique random positions
            indices = torch.randperm(total_elements)[:non_zero_count]
            
            # Convert flat indices to 2D indices
            rows = indices // n
            cols = indices % n
            
            # Set the corresponding positions to True
            tensor[rows, cols] = True

        return tensor.bool()
    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.total_steps+=1
        scale=self.warmup.get_dr(self.total_steps)
        # print("scales:",scale,self.update_proj_gap)
        for group in self.param_groups:
                    
            # if "rank" in group:
            #     self.update_proj_gap=group["update_proj_gap"]

            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad


                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if "exp_avg" not in state:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                    state["m_norm_t"]=0
                    state["v_norm_t"]=0
                    state['m_max_t']=0
                    # state['m_min_t']=0
                    # # state["c_norm_t"]=0

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                max_gradient=torch.max(grad.abs())
                # min_gradient=torch.min(grad)
                m_max_t=state["m_max_t"]
                # m_min_t=state['m_min_t']

                state['step'] += 1


                m_max_t = self.theta* m_max_t + (1 - self.theta) * max_gradient
                # m_min_t = self.theta* m_min_t + (1 - self.theta) * min_gradient
                
                m_max_hat = m_max_t / (1 - self.theta**state['step'])
                # m_min_hat = m_min_t / (1 - self.theta**state['step'])
                
                mask=grad.abs()>m_max_hat
           
                if mask.sum()>0:
                    grad[mask]=grad[mask]/max_gradient*m_max_hat
                # mask=(grad**2)>5000*exp_avg_sq
                # if state['step']>20:
                #     if mask.sum()>0:
                #         grad[mask]=grad[mask].sign()*torch.sqrt(exp_avg_sq[mask]*5000)

                state["m_max_t"]=m_max_t
                # state["m_min_t"]=m_min_t
                # ###### clipping
                grad_norm=torch.norm(grad)
                ####norm scaling
                m_norm_t,v_norm_t=state["m_norm_t"],state["v_norm_t"]
                # print("m_norm_t",m_norm_t,grad_norm)
                m_norm_t = self.gamma1 * scale*m_norm_t + (1 - self.gamma1*scale) * grad_norm
                
                v_norm_t = self.gamma2 * v_norm_t + (1 - self.gamma2) * grad_norm**2
                
                m_norm_hat = m_norm_t / (1 - (self.gamma1*scale)**state['step'])
                v_norm_hat = v_norm_t / (1 - self.gamma2**state['step'])

                c_norm_t=m_norm_hat/(torch.sqrt(v_norm_hat)+group["eps"])
                # print("grad_nrom",grad_norm,"c_norm",c_norm_t,"st",s_t,m_norm_t)

                grad=grad/grad_norm*c_norm_t
              

                # print(m_norm_t)
                state["m_norm_t"],state["v_norm_t"]=m_norm_t,v_norm_t


                ###############################norm scaling end#########################
                if self.update_proj_gap!=0:
                    if (self.total_steps % self.update_proj_gap == 0 ):
                        state["exp_avg"] = torch.zeros_like(grad)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                        # self.update_masks()
                        state['step'] = 1
                        # print("Mask Update",flush=True)


                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                beta1=beta1*scale

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1


                norm_grad=exp_avg/denom

                # else:
                grad=norm_grad
                p.add_(grad, alpha=-step_size)
        return loss
    def update_masks(self):
        overlap_ratio=0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "rank" in group:
                    assert len(p.data.shape) == 2
                    new_mask, overlap_ratio = self.update_mask_random(group['rank'], p, state['mask'])
                    state['mask'] = new_mask
                    p.mask=new_mask
        print(f"Mask overlap ratio: {overlap_ratio:.2f}")

    def update_mask(self, density, p, old_mask,sampling=False):
        if self.updating_mask_method=="grad_max":
            gradients=p.grad
        elif self.updating_mask_method=='weight_max':
            gradients=p.data
        state=self.state[p]
        m, n = gradients.shape
        total_elements = m * n
        non_zero_count = int(density * total_elements)

        # Ensure non_zero_count is within valid range
        non_zero_count = min(non_zero_count, total_elements)

        # Create a tensor with all False values
        new_mask = torch.zeros((m, n), dtype=torch.bool).to(gradients.device)

        # Calculate the absolute values of the gradients
        gradient_abs = gradients.abs()

        # Flatten the gradients to easily sort and index
        flattened_gradients = gradient_abs.view(-1)
        if sampling:

            # Step 2: Flatten the magnitudes for processing
            flattened_magnitudes = flattened_gradients

            # Step 3: Convert the magnitudes to probabilities using the softmax function
            probabilities = torch.nn.functional.softmax(flattened_magnitudes, dim=0)

            # Step 4: Determine the number of samples (50% of the data points)
            num_samples = non_zero_count

            # Step 5: Sample data points according to the probabilities
            selected_indices = torch.multinomial(probabilities, num_samples, replacement=False)

            # Step 6: Create a mask with the same shape as the flattened gradient tensor
            mask_flattened = torch.zeros_like(flattened_magnitudes, dtype=torch.bool)
            mask_flattened[selected_indices] = True

            # Reshape the mask to the original gradient shape
            new_mask = mask_flattened.view(gradients.shape)
        else:
            # Get the indices of the top non_zero_count elements
            top_indices = torch.topk(flattened_gradients, non_zero_count).indices

            # Convert the flattened indices back to 2D indices
            rows = top_indices // n
            cols = top_indices % n

            # Set the selected elements to True
            new_mask[rows, cols] = True

        # Calculate the overlap ratio
        new_mask=new_mask.bool()
        intersection_mask=new_mask & old_mask
        overlap_count = intersection_mask.sum().item()
        overlap_ratio = overlap_count / non_zero_count
        
        
        exp_avg = torch.zeros_like(state['exp_avg'])
        # Exponential moving average of squared gradient values
        exp_avg_sq = torch.zeros_like(state['exp_avg'])
        exp_avg[intersection_mask[new_mask]] = state['exp_avg'][intersection_mask[old_mask]]
        exp_avg_sq[intersection_mask[new_mask]] = state['exp_avg_sq'][intersection_mask[old_mask]]
        state['exp_avg']=exp_avg
        state['exp_avg_sq']=exp_avg_sq
        return new_mask, overlap_ratio

    def update_mask_random(self, density, p, old_mask):
        m, n = p.data.shape
        total_elements = m * n
        state=self.state[p]
        non_zero_count = int(density * total_elements)

        new_mask=torch.rand(p.data.shape).cuda() < density
        # Calculate the overlap ratio
        
        overlap_count = (new_mask & old_mask).sum().item()

                # Calculate the overlap ratio
        intersection_mask=new_mask & old_mask
        overlap_count = intersection_mask.sum().item()
        overlap_ratio = overlap_count / non_zero_count
        
        
        exp_avg = torch.zeros_like(p.data[new_mask])
        # Exponential moving average of squared gradient values
        exp_avg_sq = torch.zeros_like(p.data[new_mask])
        exp_avg[intersection_mask[new_mask]] = state['exp_avg'][intersection_mask[old_mask]]
        exp_avg_sq[intersection_mask[new_mask]] = state['exp_avg_sq'][intersection_mask[old_mask]]
        state['exp_avg']=exp_avg
        state['exp_avg_sq']=exp_avg_sq


        overlap_ratio = overlap_count / non_zero_count

        return new_mask, overlap_ratio

