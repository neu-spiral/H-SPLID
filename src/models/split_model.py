import numpy as np
import torch
import torch.nn as nn
# Imports for our backbone models
from sklearn.metrics import accuracy_score
from src.models import StandardClassificationModel
from src.hsic import hsic_split_objective
from typing import Dict, Any

class SplitTrain(StandardClassificationModel):
   
    def __init__(
        self, 
        backbone_model: nn.Module,
        **kwargs,
    ):
        super().__init__(backbone_model=backbone_model, **kwargs)
        try:
            backbone_model.feature_hook_fn
        except AttributeError:
            backbone_model.feature_hook_fn = lambda x: x
            
        self.model = backbone_model
        self.kwargs = kwargs
        self.beta_step = self.kwargs["split_hps"]["beta_step"]
        self.beta_update_step_fraction = self.kwargs["split_hps"]["beta_update_step_fraction"]
        self.shared_space_variation = self.kwargs["split_hps"]["shared_space_variation"]
        self.beta_warmup_epochs = self.kwargs["split_hps"]["beta_warmup_epochs"] if 'beta_warmup_epochs' in self.kwargs["split_hps"] else 0
        self.num_classes = self.kwargs['dataset']['n_classes']
        self.sample_frac = self.kwargs['split_hps']['beta_init_sample_fraction']
        if "resnet" in self.kwargs['model'].lower():
            # resnet
            D = backbone_model.fc.in_features
        elif "lenet" in self.kwargs['model'].lower():
            # lenet
            D = backbone_model.last_hidden_dim
        else:
            raise ValueError(f"Model {self.kwargs['model']} not supported")
        
        self.hard_beta_threshold = 1
        self.include_hsic = 0
        self.betas_per_epoch = []
        self.len_dataloader = None
        self.batch_counter = 0
        self.samples_counter = 0 # used to update betas
        self._skip_init_split = False
        
        self.register_buffer('cs_centers', torch.zeros(self.num_classes, D))
        self.register_buffer('adv_center', torch.zeros(1, D))
        self.register_buffer('center_count', torch.zeros(self.num_classes))
        self.register_buffer('betas', torch.zeros(2, D))
        self.register_buffer('beta_cs', torch.zeros(D))
        self.register_buffer('beta_adv', torch.zeros(D))
        self.register_buffer('hard_betas', torch.zeros(2, D))
        self.register_buffer('softmax_betas_avg', torch.zeros(2, D))
        self.save_hyperparameters(ignore=['backbone_model'], logger=False)
        
    def on_fit_start(self):
        self.len_dataloader = len(self.trainer.datamodule.train_dataloader()) * self.trainer.datamodule.batch_size
        
        # only run init_split_module if the model was not loaded from a checkpoint
        if not self._skip_init_split:
            train_dl = self.trainer.datamodule.train_dataloader()  
            self.init_split_module(train_dl, sample_frac=self.sample_frac)
        
        self._initialize_feature_hook_betas()

    def on_test_start(self):
        self._initialize_feature_hook_betas()
        
    def on_validation_start(self):
        self._initialize_feature_hook_betas()
        
    def on_load_checkpoint(self, checkpoint: Dict[str,Any]) -> None:
        extras = checkpoint.get("extras", {})
        self.betas_per_epoch     = extras.get("betas_per_epoch",     [])
        self.include_hsic        = extras.get("include_hsic",        0)
        self.batch_counter       = extras.get("batch_counter",       0)
        self.samples_counter     = extras.get("samples_counter",     0)
        
        self._skip_init_split = True
        self._initialize_feature_hook_betas()
    
    
    def on_save_checkpoint(self, checkpoint: Dict[str,Any]) -> Dict[str,Any]:
        return {
            "extras": {
                "betas_per_epoch":     self.betas_per_epoch,
                "include_hsic":        self.include_hsic,
                "batch_counter":       self.batch_counter,
                "samples_counter":     self.samples_counter,
            }
        }
        
    def _initialize_feature_hook_betas(self):
        betas_cond = (self.hard_betas[0].sum().item() > self.hard_beta_threshold) and (self.hard_betas[1].sum().item() > self.hard_beta_threshold)
        hard_split = self.kwargs["split_hps"]["hard_space_split"] and betas_cond
        # warmup condition during training:
        try:
            hard_split = hard_split and self.trainer.current_epoch >= self.beta_warmup_epochs
        except RuntimeError:
            pass
        if hard_split:
            self.model.feature_hook_fn = lambda x: x * self.hard_betas[0]
        else:
            self.model.feature_hook_fn = lambda x: x * self.beta_cs
            
    def init_split_module(self, dataloader, sample_frac: float = 1.0)-> None:
        # Before start of training, init betas and centers, self.device = cpu in the beginning
        # This is only done once
        # compute beta splitting exactly
        self.eval()
        self.len_dataloader = len(dataloader) * dataloader.batch_size
        max_samples = int(sample_frac * len(dataloader.dataset))
        with torch.no_grad():
            embeddings = []
            targets = []
            
            seen = 0
            for batch in dataloader:
                # NOTE: Compute on clean data
                embeddings.append(self.model.encode(batch[1].to(self.device)).detach().cpu())
                targets.append(batch[2].detach().cpu())
                seen += batch[1].size(0)
                if seen >= max_samples:
                    break
                
            embeddings = torch.cat(embeddings, dim=0).to(self.device)
            targets = torch.cat(targets).long()
            # self.num_classes = len(set(targets.tolist()))
            # print(f"Initializing split module with {self.num_classes} classes")
            # print(f"Unique classes: {set(targets.tolist())}")
            assignment_matrix = int_to_one_hot(targets, n_integers=self.num_classes).to(self.device)
            # Initializes the center count
            # This means centroid learning rate at the beginning is scaled by a hundred
            self.center_count.copy_((torch.ones(assignment_matrix.shape[1])*100.0))
            # Initializes the centers given the labels inplace
            update_centers_(self, embeddings, assignment_matrix)
            # Compute optimal betas
            new_betas = get_beta_weights(embeddings, 
                                          [self.cs_centers, self.adv_center],
                                          embeddings.device,
                                          class_assignments=assignment_matrix).to(self.device)
            self.betas.copy_(new_betas)
            self.beta_cs.copy_(self.betas[0])
            self.beta_adv.copy_(self.betas[1])
            self.hard_betas.copy_(get_hard_assignments(self.betas.clone(), centers=[self.cs_centers, self.adv_center], shared_space_variation=self.shared_space_variation))
            # NOTE: This is only used if `update_betas_per_epoch` is True
            self.softmax_betas_avg.copy_(self.betas.clone())
            
            
            
        self.train()
        self._initialize_feature_hook_betas()
        
    def on_train_epoch_start(self) -> None:
        # setup hsic schedule
        warmup_epochs = self.kwargs["split_hps"]["hsic_warmup_epochs"]
        if not isinstance(warmup_epochs, str) and warmup_epochs > 0:
            if self.trainer.current_epoch < warmup_epochs:
                self.include_hsic = 0
            else:
                self.include_hsic = 1
        elif isinstance(warmup_epochs, str) and warmup_epochs == "on_hard_space_split":
            if self.kwargs["split_hps"]["hard_space_split"] and self.hard_betas[0].sum().item() > self.hard_beta_threshold:
                self.include_hsic = 1
            else:
                self.include_hsic = 0
        else:
            self.include_hsic = 1
        
        # logging
        self.log('split_train/cs_dim', self.hard_betas[0].sum().item())
        self.log('split_train/adv_dim', self.hard_betas[1].sum().item())
        self.log('split_train/cs_dim_soft_avg', self.beta_cs.mean().item())
        self.log('split_train/adv_dim_soft_avg', self.beta_adv.mean().item())
        d = self.beta_cs.shape[0]
        sorted_beta_cs = torch.sort(self.beta_cs, descending=True)[0]
        sorted_beta_adv = torch.sort(self.beta_adv, descending=True)[0]
        self.log('split_train/cs_dim_soft_avg_top_10_perc', sorted_beta_cs[:int(d*0.1)].mean().item())
        self.log('split_train/adv_dim_soft_avg_top_10_perc', sorted_beta_adv[:int(d*0.1)].mean().item())
        self.log('split_train/cs_dim_soft_avg_top_1_perc', sorted_beta_cs[:int(d*0.01)].mean().item())
        self.log('split_train/adv_dim_soft_avg_top_1_perc', sorted_beta_adv[:int(d*0.01)].mean().item())
    
    def on_train_epoch_end(self):
        # log betas over training for later plotting
        self.betas_per_epoch.append(self.beta_cs.detach().cpu().numpy())
    def update_betas_(self, new_betas):
        if not self.kwargs["split_hps"]["update_betas_per_epoch"]:
            self.beta_cs.copy_(self.beta_cs * self.beta_step + new_betas[0].detach() * (1-self.beta_step))
        else:
            self.softmax_betas_avg.copy_(self.softmax_betas_avg * self.beta_step + new_betas * (1-self.beta_step))
            
            if self.samples_counter >= self.beta_update_step_fraction * self.len_dataloader:
                self.beta_cs.copy_(self.beta_cs * self.beta_step + self.softmax_betas_avg[0] * (1-self.beta_step))
                self.samples_counter = 0
        self.beta_adv.copy_(1 - self.beta_cs)
        self.betas.copy_(torch.stack([self.beta_cs, self.beta_adv]))
        self.hard_betas.copy_(get_hard_assignments(self.betas.clone(), centers=[self.cs_centers, self.adv_center], shared_space_variation=self.shared_space_variation))
        # NOTE: Not sure if this is needed, but want to make sure that new betas are used
        self._initialize_feature_hook_betas()
                
    
    def compute_hsic_terms(self, z_cluster, z_shared, assignment_matrix, input_data):
        # Flatten input data
        input_data_flat = input_data.view(-1, np.prod(input_data.size()[1:]))
        # Compute split hsic objectives
        l_hsic_y_cluster, l_hsic_x_shared, l_hsic_zz_val = hsic_split_objective(
                z_cluster=z_cluster, 
                z_shared=z_shared,
                h_target=assignment_matrix.float(),
                h_data=input_data_flat,
                sigma=self.kwargs['sigma'],
                k_type_y=self.kwargs['k_type_y'],
        )
        # compute opposite terms
        l_hsic_y_shared, l_hsic_x_cluster, _ = hsic_split_objective(
                z_cluster=z_shared, 
                z_shared=z_cluster,
                h_target=assignment_matrix.float(),
                h_data=input_data_flat,
                sigma=self.kwargs['sigma'],
                k_type_y=self.kwargs['k_type_y'],
        )
        
        return l_hsic_y_cluster, l_hsic_x_shared, l_hsic_zz_val, l_hsic_y_shared, l_hsic_x_cluster
    
    def _predict(self, batch, batch_idx):
        if len(batch) == 2: # dataloader: data, target
            data_aug, target = batch
            data_orig = None
            masks = None
        elif len(batch) == 3:
            data_aug, data_orig, target = batch
            masks = None
        elif len(batch) == 4: # dataloader: data, target, masks (e.g., coco)
            data_aug, data_orig, target, masks = batch 
        else:
            raise ValueError(f"Unexpected batch size of {len(batch)} encountered")    
        
        emb = self.model.encode(data_aug)
        
        prediction = self.model.predict(emb)
            
        prediction_labels = prediction.argmax(1).detach().cpu().numpy()
        accuracy = accuracy_score(prediction_labels, target.detach().cpu().numpy())*100 
        return accuracy

    def _forward_step(self, batch, batch_idx, mode):
        if mode == 'test':
            loss = 0.0 # loss is not computed during test
            return loss, self._predict(batch, batch_idx)
        
        # count batches to infer when epoch ends
        # only increase during training
        if mode == "train":
            self.batch_counter += 1
            self.samples_counter += batch[1].size(0)
        if len(batch) == 2:
            data_aug, target = batch
            data_orig = None
            masks = None
        elif len(batch) == 3:
            data_aug, data_orig, target = batch
            masks = None
        elif len(batch) == 4: # dataloader: data, target, masks (e.g., coco)
            data_aug, data_orig, target, masks = batch 
        else:
            raise ValueError(f"Unexpected batch size of {len(batch)} encountered")

        emb = self.model.encode(data_aug)
        assignment_matrix = int_to_one_hot(target, self.num_classes)
        
        if mode == "train":
            update_centers_(self, emb, assignment_matrix)
        # compute split loss
        softmax_betas, l_cluster, l_shared = acedec_loss(emb, assignment_matrix, self)
        if mode == "train":
            # update betas inplace
            self.update_betas_(softmax_betas)


        # create split embeddings
        betas_cond = (self.hard_betas[0].sum().item() > self.hard_beta_threshold) and (self.hard_betas[1].sum().item() > self.hard_beta_threshold)
        # warmup condition:
        betas_cond = betas_cond and self.trainer.current_epoch >= self.beta_warmup_epochs
        if self.kwargs["split_hps"]["hard_space_split"] and betas_cond:
            z_cluster = emb[:, self.hard_betas[0].int().bool()]
            z_shared = emb[:, self.hard_betas[1].int().bool()]
        else:
            z_cluster, z_shared = split_embeddings(emb, self.beta_cs, shared_space_variation=self.shared_space_variation)
                
        # compute hsic terms
        l_hsic_y_cluster, l_hsic_x_shared, l_hsic_zz_val, l_hsic_y_shared, l_hsic_x_cluster = self.compute_hsic_terms(z_cluster, z_shared, assignment_matrix, batch[0])
        # maximize label information in cluster space
        l_cluster_hsic = self.kwargs["split_hps"]['lambda_cluster'] * l_cluster - self.include_hsic*self.kwargs["split_hps"]['lambda_y_cluster'] * l_hsic_y_cluster
        # maximize input information in shared space
        l_shared_hsic = self.kwargs["split_hps"]['lambda_shared'] * l_shared - self.include_hsic*self.kwargs["split_hps"]['lambda_x_shared'] * l_hsic_x_shared
        if self.kwargs["split_hps"]["additional_hsic_y_x"]:
            # minimize input information in cluster space
            l_cluster_hsic += self.include_hsic*self.kwargs["split_hps"]['l_cluster_hsicx_weight'] * l_hsic_x_cluster
            # minimize label information in shared space
            l_shared_hsic += self.include_hsic*self.kwargs["split_hps"]["l_shared_hsicy_weight"] * l_hsic_y_shared
        if self.kwargs["split_hps"]["additional_hsic_z_z"]:
            # minimize information between z_cluster and z_shared
            l_cluster_hsic += self.include_hsic*self.kwargs["split_hps"]["l_hsic_z_z_weight"]*l_hsic_zz_val
        

        prediction = self.model.predict(emb)
        
        split_loss = l_cluster_hsic + l_shared_hsic
        ce_loss = self.compute_loss(prediction, target)
        loss = split_loss + ce_loss * self.kwargs["split_hps"]['cross_entropy_weight']
        prediction_labels = prediction.argmax(1).detach().cpu().numpy()
        accuracy =  accuracy_score(prediction_labels, target.detach().cpu().numpy())*100
        
        # reset batch counter if end of epoch is reached
        if self.batch_counter == self.len_dataloader:
            self.batch_counter = 0
        
        # Logging
        self.log(f'split_{mode}/Cluster_Loss', l_cluster)
        self.log(f'split_{mode}/Shared_Loss', l_shared)
        self.log(f'split_{mode}/Split_loss', split_loss)
        self.log(f'split_{mode}/hsic(y, z_cluster) inc.', l_hsic_y_cluster)
        self.log(f'split_{mode}/hsic(x, z_shared) inc.', l_hsic_x_shared)
        self.log(f'split_{mode}/hsic(x, z_cluster) dec.', l_hsic_x_cluster)
        self.log(f'split_{mode}/hsic(y, z_shared) dec.', l_hsic_y_shared)
        self.log(f'split_{mode}/hsic(z_cluster, z_shared) dec.', l_hsic_zz_val)
        self.log(f'{mode}/ce_loss', ce_loss)
            
        return loss, accuracy
    
@torch.no_grad()
def compute_centers(centers, embedded, assignment_matrix, count, center_lr=0.5):
    device = embedded.device
    # Minibatch variant of DCN update
    copy_count = count.clone().unsqueeze(1).to(device)
    batch_cluster_sums = torch.einsum('nd,nc->cd',
                                      embedded.detach(),
                                      assignment_matrix.to(embedded.dtype))
    mask_sum = assignment_matrix.sum(0).unsqueeze(1).to(embedded.dtype)
    nonzero_mask = (mask_sum.squeeze(1) != 0).to(device)
    copy_count[nonzero_mask] = (1 - center_lr) * copy_count[nonzero_mask]
    copy_count[nonzero_mask] += center_lr * mask_sum[nonzero_mask]
    per_center_lr = 1.0 / (copy_count[nonzero_mask] + 1)
    batch_centers = batch_cluster_sums[nonzero_mask] / mask_sum[nonzero_mask]

    if torch.all(centers == 0):
        new_centers = batch_centers
    else:
        new_centers = centers.clone()
        new_centers[nonzero_mask] = (
            (1.0 - per_center_lr) * centers[nonzero_mask]
            + per_center_lr * batch_centers
        )

    return new_centers, copy_count.squeeze(1)

def update_centers_(model, embedding, assignment_matrix, update_step=0.5):
    cs_centers_moving_avg, center_count = compute_centers(model.cs_centers, embedding, assignment_matrix,
                                                          model.center_count, center_lr=update_step)
    # update adversarial space center
    adv_center_moving_avg = model.adv_center * (1-update_step) + embedding.detach().mean(0).unsqueeze(0) * update_step
    
    model.cs_centers.copy_(cs_centers_moving_avg)
    model.adv_center.copy_(adv_center_moving_avg)
    model.center_count.copy_(center_count)
    
def compression_loss(embedded, centers, weights=None, assignment_matrix=None)->torch.Tensor:
    dist = squared_euclidean_distance(embedded, centers, weights=weights)
    if assignment_matrix is None:
        loss = (dist.min(dim=1)[0]).mean()#.sum()/(embedded.shape[0]*embedded.shape[1])
    else:
        loss = (dist * assignment_matrix).mean()#.sum()/(embedded.shape[0]*embedded.shape[1])
    return loss    

def acedec_loss(embedding, assignment_matrix, model):
    device = embedding.device
    # How to update betas? Exact computation or learned?
    centers = [model.cs_centers, model.adv_center]
    # compute betas
    softmax_betas = get_beta_weights(embedding.detach(), centers, embedding.device, class_assignments=assignment_matrix).to(device)
    # compute losses
    cluster_space_loss = compression_loss(embedding, centers=model.cs_centers,  weights=softmax_betas[0], assignment_matrix=assignment_matrix)
    adv_space_loss = compression_loss(embedding, centers=model.adv_center, weights=softmax_betas[1])
    
    return softmax_betas, cluster_space_loss, adv_space_loss

def optimal_beta(kmeans_loss: torch.Tensor, other_losses_mean_sum: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    """
    Calculate optimal values for the beta weight for each dimension.
    
    Parameters
    ----------
    kmeans_loss: torch.Tensor
        a 1 x d vector of the kmeans losses per dimension.
    other_losses_mean_sum: torch.Tensor
        a 1 x d vector of the kmeans losses of all other clusterings except the one in 'kmeans_loss'.
    
    Returns
    -------
    optimal_beta_weights: torch.Tensor
        a 1 x d vector containing the optimal weights for the softmax to indicate which dimensions are important for each clustering.
        Calculated via -torch.log(kmeans_loss/other_losses_mean_sum)
    """
    return -torch.log(kmeans_loss / (other_losses_mean_sum + eps))

def squared_euclidean_distance(tensor1: torch.Tensor, tensor2: torch.Tensor,
                               weights: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate the pairwise squared Euclidean distance between two tensors.
    Each row in the tensors is interpreted as a separate object, while each column represents its features.
    Therefore, the result of an (4x3) and (12x3) tensor will be a (4x12) tensor.
    Optionally, features can be individually weighted.
    The default behavior is that all features are weighted by 1.

    Parameters
    ----------
    tensor1 : torch.Tensor
        the first tensor
    tensor2 : torch.Tensor
        the second tensor
    weights : torch.Tensor
        tensor containing the weights of the features (default: None)

    Returns
    -------
    squared_diffs : torch.Tensor
        the pairwise squared Euclidean distances
    """
    assert tensor1.shape[1] == tensor2.shape[1], "The number of features of the two input tensors must match."
    ta = tensor1.unsqueeze(1)
    tb = tensor2.unsqueeze(0)
    squared_diffs = (ta - tb)
    if weights is not None:
        assert tensor1.shape[1] == weights.shape[0]
        weights_unsqueezed = weights.unsqueeze(0).unsqueeze(1)
        squared_diffs = squared_diffs * weights_unsqueezed
    squared_diffs = squared_diffs.pow(2).sum(2)
    return squared_diffs


def int_to_one_hot(int_tensor: torch.Tensor, n_integers: int) -> torch.Tensor:
    """
    Convert a tensor containing integers (e.g. labels) to an one hot encoding.
    Here, each integer gets its own features in the resulting tensor, where only the values 0 or 1 are accepted.
    E.g. [0,0,1,2,1] gets
    [[1,0,0],
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [0,1,0]]

    Parameters
    ----------
    int_tensor : torch.Tensor
        The original tensor containing integers
    n_integers : int
        The number of different integers within int_tensor

    Returns
    -------
    onehot : torch.Tensor
        The final one hot encoding tensor
    """
    onehot = torch.zeros([int_tensor.shape[0], n_integers], dtype=torch.float, device=int_tensor.device)
    onehot.scatter_(1, int_tensor.unsqueeze(1).long(), 1)
    return onehot

def calculate_optimal_beta_weights_special_case(data: torch.Tensor, centers: list, device: str, class_assignments=None, batch_size=1024) -> torch.Tensor:
    """
    The beta weights have a closed form solution if we have two subspaces, so the optimal values given the data, centers and V can be computed.
    See supplement of Details (Don't) Matter: Isolating Cluster Information in Deep Embedded Spaces. IJCAI 2021: 2826


    Parameters
    ----------
    data : torch.Tensor
        input data
    centers : list
        list of torch.Tensor, cluster centers for each clustering
    batch_size : int
        size of the data batches (default: 32)

    Returns
    -------
    optimal_beta_weights: torch.Tensor
        a c x d vector containing the optimal weights for the softmax to indicate which dimensions d are important for each clustering c.
    """
    with torch.no_grad():
        # Pre-allocate km_losses accumulators for each clustering
        km_losses = [torch.zeros(data.shape[1], device='cpu') for _ in centers]
        
        # Process data in batches
        num_batches = (data.shape[0] + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, data.shape[0])
            batch_data = data[start_idx:end_idx].to(device)
            
            # If class_assignments is provided, get the corresponding batch
            batch_assignments = None
            if class_assignments is not None:
                batch_assignments = class_assignments[start_idx:end_idx]
            
            # Process each clustering
            for i, centers_i in enumerate(centers):
                centers_i = centers_i.to(device)
                
                # Calculate distances
                weighted_squared_diff = squared_euclidean_distance(
                    batch_data.unsqueeze(1), 
                    centers_i.unsqueeze(1)
                )
                
                # Get assignments if not provided
                if batch_assignments is None:
                    assignments = weighted_squared_diff.detach().sum(2).argmin(1)
                    if len(set(assignments.tolist())) > 1:
                        one_hot_mask = int_to_one_hot(assignments, centers_i.shape[0])
                        weighted_squared_diff_masked = weighted_squared_diff * one_hot_mask.unsqueeze(2)
                    else:
                        # Handle single cluster case
                        weighted_squared_diff_masked = weighted_squared_diff
                else:
                    # Use the provided batch assignments
                    weighted_squared_diff_masked = weighted_squared_diff * batch_assignments.unsqueeze(2)
                
                # Sum up the batch contribution
                batch_km_loss = weighted_squared_diff_masked.sum(0).sum(0).detach().cpu()
                
                # Accumulate to the total loss
                km_losses[i] += batch_km_loss
                
                # Free GPU memory
                centers_i = centers_i.cpu()
        
        # Calculate beta_weights for each dimension and clustering based on kmeans losses
        best_weights = []
        best_weights.append(optimal_beta(km_losses[0], km_losses[1]))
        best_weights.append(optimal_beta(km_losses[1], km_losses[0]))
        best_weights = torch.stack(best_weights)
        
    return best_weights

def get_beta_weights(data: torch.Tensor, centers: list, device: str, class_assignments: torch.Tensor) -> torch.Tensor:
    optimal_weights = calculate_optimal_beta_weights_special_case(data, centers, device, class_assignments)
    soft_betas_cs = torch.nn.functional.softmax(optimal_weights, dim=0)[0]
    # collapsed dimensions will be assigned to the adv space
    soft_betas_cs[soft_betas_cs.isnan()] = 1e-12
    soft_beta_adv = 1 - soft_betas_cs
    soft_betas = torch.stack([soft_betas_cs, soft_beta_adv])
    return soft_betas

def beta_weights_init(P: list, n_dims: int, high_value: float = 0.9) -> torch.Tensor:
    """
    Initializes parameters of the softmax such that betas will be set to high_value in dimensions which form a cluster subspace according to P
    and set to (1 - high_value)/(len(P) - 1) for the other clusterings.
    
    Parameters
    ----------
    P : list
        list containing projections for each subspace
    n_dims : int
        dimensionality of the embedded data
    high_value : float
        value that should be initially used to indicate strength of assignment of a specific dimension to the clustering (default: 0.9)
    
    Returns
    ----------
    beta_weights : torch.Tensor
        initialized weights that are input in the softmax to get the betas.
    """
    weight_high = 1.0
    n_sub_clusterings = len(P)
    beta_hard = np.zeros((n_sub_clusterings, n_dims), dtype=np.float32)
    for sub_i, p in enumerate(P):
        for dim in p:
            beta_hard[sub_i, dim] = 1.0
    weight_high_exp = np.exp(weight_high)
    # Because high_value = weight_high/(weight_high +low_classes*weight_low)
    n_low_classes = len(P) - 1
    weight_low_exp = weight_high_exp * (1.0 - high_value) / (high_value * n_low_classes)
    weight_low = np.log(weight_low_exp)
    beta_soft_weights = beta_hard * (weight_high - weight_low) + weight_low
    return torch.tensor(beta_soft_weights, dtype=torch.float32)

def get_hard_assignments(betas: torch.Tensor, centers: list | None, shared_space_variation: float = 0.025, shared_space_idx=None) -> float:
    """
    Converts the softmax betas back to hard assignments P and returns them as a list.

    Parameters
    ----------
    betas : torch.Tensor
        c x d soft assignment weights matrix for c clusterings and d dimensions.
    centers : list
        list of torch.Tensor, cluster centers for each clustering
    shared_space_variation : float
        specifies how much beta in the shared space is allowed to diverge from the uniform distribution. Only needed if a shared space (space with one cluster) exists (default: 0.05)

    Returns
    ----------
    P : list
        list containing indices for projections for each clustering
    """
    betas_hard = betas.clone()
    # Check if a shared space with a single cluster center exist
    if shared_space_idx is None:
        shared_space_idx = [i for i, centers_i in enumerate(centers) if centers_i.shape[0] == 1]
    if shared_space_idx:
        # Specifies how much beta in the shared space is allowed to diverge from the uniform distribution
        if isinstance(shared_space_idx, list):
            shared_space_idx = shared_space_idx[0]
        equal_threshold = 1.0 / betas.shape[0]
        # Increase Weight of shared space dimensions that are close to the uniform distribution
        equal_threshold -= shared_space_variation
        betas[shared_space_idx][betas[shared_space_idx] > equal_threshold] += 1

    # Select highest assigned dimensions to P
    max_assigned_dims = betas.argmax(0)
    P = [[] for _ in range(betas.shape[0])]
    for dim_i, cluster_subspace_id in enumerate(max_assigned_dims):
        P[cluster_subspace_id].append(dim_i)
    betas_hard[0, P[0]] = 1
    betas_hard[1, P[0]] = 0
    betas_hard[0, P[1]] = 0
    betas_hard[1, P[1]] = 1
    
    return betas_hard

def split_embeddings(embs, beta_cs, hard_split=False, shared_space_variation=0.025):
    if hard_split:
        _soft_betas = torch.stack([beta_cs, 1 - beta_cs]).clone()
        hard_betas = get_hard_assignments(_soft_betas, centers=None, shared_space_idx=[1], shared_space_variation=shared_space_variation)
        z_cluster = embs[:, hard_betas[0].int().bool()]
        z_shared = embs[:, hard_betas[1].int().bool()]
    else:
        beta_adv = 1.0 - beta_cs
        z_cluster = embs * beta_cs
        z_shared = embs * beta_adv 
    return z_cluster, z_shared