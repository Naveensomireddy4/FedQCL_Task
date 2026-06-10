import torch

def calculate_percentage_difference(client_weights, server_weights):
    # Check if client_weights and server_weights are dictionaries
    if not isinstance(client_weights, dict) or not isinstance(server_weights, dict):
        raise TypeError("Both client_weights and server_weights must be dictionaries. Received: "
                        f"client_weights={type(client_weights)}, server_weights={type(server_weights)}")

    total_perc_diff = 0
    num_weights = 0
    for key in server_weights.keys():
        server_weight = server_weights[key]
        client_weight = client_weights[key]
        
        # Calculate percentage difference
        perc_diff = torch.norm(server_weight - client_weight) / torch.norm(server_weight)
        
        total_perc_diff += perc_diff.item()
        num_weights += 1
    
    # Calculate mean percentage difference
    if num_weights > 0:
        mean_perc_diff = total_perc_diff / num_weights
    else:
        mean_perc_diff = 0
    
    return mean_perc_diff


def calculate_similarity(client_weights, server_weights):
    # Check if client_weights and server_weights are dictionaries
    if not isinstance(client_weights, dict) or not isinstance(server_weights, dict):
        raise TypeError("Both client_weights and server_weights must be dictionaries. Received: "
                        f"client_weights={type(client_weights)}, server_weights={type(server_weights)}")

    similarity = {}
    for key in server_weights.keys():
        server_weight = server_weights[key]
        client_weight = client_weights[key]
        
        # Flatten the tensors for similarity calculation
        server_flat = server_weight.view(-1)
        client_flat = client_weight.view(-1)
        
        # Cosine similarity
        cosine_sim = torch.dot(server_flat, client_flat) / (torch.norm(server_flat) * torch.norm(client_flat))
        
        similarity[key] = cosine_sim.item()
    
    return similarity

def calculate_distance(client_weights, server_weights):
    # Check if client_weights and server_weights are dictionaries
    if not isinstance(client_weights, dict) or not isinstance(server_weights, dict):
        raise TypeError("Both client_weights and server_weights must be dictionaries. Received: "
                        f"client_weights={type(client_weights)}, server_weights={type(server_weights)}")

    distance = {}
    for key in server_weights.keys():
        server_weight = server_weights[key]
        client_weight = client_weights[key]
        
        # Euclidean distance
        euclidean_dist = torch.norm(server_weight - client_weight)
        
        # Manhattan distance
        manhattan_dist = torch.norm(server_weight - client_weight, p=1)
        
        distance[key] = {'euclidean': euclidean_dist.item(), 'manhattan': manhattan_dist.item()}
    
    return distance
