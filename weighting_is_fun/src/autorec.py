import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class _AutoRecModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=500, dropout=0.05):
        super(_AutoRecModel, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Sigmoid()  # Sigmoid often works better for explicit reconstruction
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.activation(x)
        x = self.decoder(x)
        return x

    @staticmethod
    def weighted_mse_loss(output, target, weights):
        # The core magic: Weights are applied to the squared error
        loss = weights * (output - target) ** 2
        return loss.sum()


class AutoRec:
    """
    AutoRec model wrapper.
    Trains an Item-based AutoEncoder with weighted MSE loss.
    """
    def __init__(self, unobserved_weight=0.1, hidden_dim=200, dropout=0.05, lr=1e-3, epochs=20, batch_size=256, weight_decay=1e-4):
        self.unobserved_weight = unobserved_weight
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.predictions = None

    def fit(self, matrix, show_progress=False):
        # I-AutoRec: Treat Items as samples, Users as dimensions.
        # matrix is (Users, Items). Transpose to (Items, Users).
        X = matrix.T.tocsr() 
        
        # Targets: Binary preference (Implicit logic)
        data_binary = (X > 0).astype(np.float32)
        # Weights: The provided confidence values
        weights = X.astype(np.float32)
        
        num_items, num_users = X.shape 
        
        # Initialize Model (input dim = num_users)
        self.model = _AutoRecModel(num_users, self.hidden_dim, self.dropout).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Training Loop
        self.model.train()
        n_samples = num_items
        indices = np.arange(n_samples)
        
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            epoch_loss = 0
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]
                
                batch_data = torch.FloatTensor(data_binary[batch_idx].toarray()).to(self.device)
                
                # Handle weights
                w_array = weights[batch_idx].toarray()

                # CRITICAL: Prevent the unobserved weight from being too negligible
                w_array[w_array == 0] = self.unobserved_weight

                batch_weights = torch.FloatTensor(w_array).to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_data)
                
                loss = _AutoRecModel.weighted_mse_loss(output, batch_data, batch_weights)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if show_progress:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}")

        # Precompute predictions for all users
        self.model.eval()
        with torch.no_grad():
            all_data = torch.FloatTensor(data_binary.toarray()).to(self.device)
            # Output shape: (Items, Users)
            preds = self.model(all_data) 
            # Transpose to (Users, Items) for easy per-user lookup
            self.predictions = preds.detach().cpu().numpy().T

    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True, filter_items=None, recalculate_user=False):
        if self.predictions is None:
            raise ValueError("Model must be fit before recommending.")
            
        # Helper to process a single user row
        def get_top_k(u_idx, u_history):
            scores = self.predictions[u_idx]
            if filter_already_liked_items and len(u_history.indices) > 0:
                scores[u_history.indices] = -np.inf
            k = min(N, len(scores))
            # Optimization: argpartition is faster than sort for finding top k
            best_ids = np.argpartition(scores, -k)[-k:]
            best_scores = scores[best_ids]
            sorted_idx = np.argsort(best_scores)[::-1]
            return best_ids[sorted_idx], best_scores[sorted_idx]

        # Handle batch or scalar inputs
        if isinstance(userid, (int, np.integer)):
            return get_top_k(userid, user_items)
        else:
            batch_size = len(userid)
            ids_batch = np.zeros((batch_size, N), dtype=np.int32)
            scores_batch = np.zeros((batch_size, N), dtype=np.float32)
            for i, u_id in enumerate(userid):
                ids, scores = get_top_k(u_id, user_items[i])
                count = len(ids)
                ids_batch[i, :count] = ids
                scores_batch[i, :count] = scores
            return ids_batch, scores_batch
