import os
import torch
import numpy as np

from env_wrapper import BelotAECEnv
from model import RecurrentPPOModel

# --- Formatting Helpers ---
SUITS = ['♣ (Clubs)', '♦ (Diamonds)', '♥ (Hearts)', '♠ (Spades)']
RANKS = ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']

def decode_card(card_id):
    suit = SUITS[card_id // 8]
    rank = RANKS[card_id % 8]
    return f"{rank} of {suit}"

def decode_hand(hand):
    if not hand:
        return "Empty"
    return ", ".join([decode_card(c) for c in sorted(hand)])

def decode_action(action):
    if action < 32:
        return f"Plays: {decode_card(action)}"
    elif action == 32:
        return "Passes"
    elif action == 33:
        return "Accepts Face-Up Suit"
    elif 34 <= action <= 37:
        return f"Picks Suit: {SUITS[action - 34]}"
    return f"Unknown Action ({action})"

def load_model(path):
    print(f"Loading checkpoint: {path}")
    model = RecurrentPPOModel()
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        raise FileNotFoundError(f"Could not find {path}. Did you save it?")
    return model

def visualize_match():
    # 1. Load the two generational models (Running on CPU for single-step inference)
    print("=== BELOT AI SHOWDOWN ===")
    print("Team 0 (Players 0 & 2): Epoch 1000 Model")
    print("Team 1 (Players 1 & 3): Epoch 500 Model\n")
    
    model_1000 = load_model("checkpoints/model_epoch_1000.pt")
    model_500 = load_model("checkpoints/model_epoch_500.pt")
    
    # Map agents to their respective brain
    brains = {
        "player_0": model_1000,
        "player_1": model_500,
        "player_2": model_1000,
        "player_3": model_500
    }

    env = BelotAECEnv()
    
    game_number = 1
    
    # Play until a team breaks 101 points
    while max(env.match_scores) < 101:
        print(f"\n{'='*40}")
        print(f"       STARTING HAND {game_number}")
        print(f"  SCORE -> Team 0: {env.match_scores[0]} | Team 1: {env.match_scores[1]}")
        print(f"{'='*40}")
        
        env.reset()
        
        # Track state to cleanly print events exactly once
        current_phase = "BIDDING"
        tricks_printed = -1 
        
        # Initialize Hidden States for the recurrent network
        hidden_states = {
            agent: (torch.zeros(1, 1, 256), torch.zeros(1, 1, 256)) 
            for agent in env.possible_agents
        }
        
        print(f"Dealer is Player {env.belot.dealer}")
        if env.belot.face_up_card is not None:
            print(f"Face-Up Card: {decode_card(env.belot.face_up_card)}")
        print("-" * 40)

        # Loop through the AEC cycle for 1 Hand (Episode)
        for agent in env.agent_iter():
            obs_dict, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                env.step(None) # Dead step required by PettingZoo
                continue
                
            abs_id = int(agent[-1])
            
            # --- DISPLAY LOGIC ---
            # Phase Transition
            if env.belot.phase == "PLAYING" and current_phase == "BIDDING":
                print("\n>>> BIDDING COMPLETE <<<")
                trump_suit = SUITS[env.belot.trump] if env.belot.trump is not None else "None"
                print(f"Declarer: Player {env.belot.declarer} | Trump: {trump_suit}")
                print("-" * 40)
                current_phase = "PLAYING"

            # Start of a new trick (Print hands)
            if env.belot.phase == "PLAYING" and env.belot.tricks_played > tricks_printed:
                print(f"\n--- TRICK {env.belot.tricks_played + 1} ---")
                for i in range(4):
                    print(f"Player {i} Hand: {decode_hand(env.belot.hands[i])}")
                print("")
                tricks_printed = env.belot.tricks_played

            # --- AI DECISION LOGIC ---
            obs = torch.tensor(obs_dict["observation"], dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(obs_dict["action_mask"], dtype=torch.float32).unsqueeze(0)
            hc = hidden_states[agent]
            
            # Forward pass through the specific agent's model
            model = brains[agent]
            with torch.no_grad():
                dist, _, new_hc = model(obs, hc, mask, is_sequence=False)
                # We use .sample() for stochastic play, but if you want to see their 
                # absolute best move, you could use torch.argmax(dist.probs)
                action = dist.sample().item()
            
            # Print Action
            print(f"Player {abs_id} {decode_action(action)}")
            
            # Step Env & Update HC
            env.step(action)
            hidden_states[agent] = new_hc
            
            # Trick Resolution Print
            if env.belot.phase == "PLAYING" and len(env.belot.current_trick) == 0 and env.belot.tricks_played > tricks_printed:
                # The environment resolves the trick internally before the next player acts
                winner = env.belot.trick_history[-1][0][0] # Trick history holds the winning info internally if evaluated
                # Actually, env.py just advances to the winner. So current_player IS the winner.
                print(f"*** Player {env.belot.current_player} takes the trick! ***")

        # End of Hand Summary
        print(f"\n{'='*40}")
        print(f"HAND {game_number} COMPLETE")
        print(f"Raw Points -> Team 0: {env.belot.raw_points_by_team[0]} | Team 1: {env.belot.raw_points_by_team[1]}")
        
        # Check for Bolts
        bolt_us = env.belot.bolts_by_team[0]
        bolt_them = env.belot.bolts_by_team[1]
        print(f"Bolts      -> Team 0: {bolt_us} | Team 1: {bolt_them}")
        
        game_number += 1

    # Match Complete
    print(f"\n\n🏆 MATCH FINISHED 🏆")
    print(f"FINAL SCORE -> Team 0: {env.match_scores[0]} | Team 1: {env.match_scores[1]}")
    if env.match_scores[0] > env.match_scores[1]:
        print("WINNER: Team 0 (Epoch 1000 Model)")
    else:
        print("WINNER: Team 1 (Epoch 500 Model)")

if __name__ == "__main__":
    visualize_match()