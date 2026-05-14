import numpy as np

class BelotEnv:
    def __init__(self):
        # Action Space: 0-31 for playing cards, 32=Pass, 33=Accept, 34-37=Pick Suit
        self.action_space_size = 38
        self.num_players = 4
        
        # Internal state
        self.dealer = 0
        self.reset()

    def reset(self):
        # Deck setup: 32 cards. IDs 0-31
        # Suit = id // 8, Rank = id % 8 (0=7, 1=8, 2=9, 3=10, 4=J, 5=Q, 6=K, 7=A)
        self.deck = np.random.permutation(32).tolist()
        
        self.hands = [[] for _ in range(self.num_players)]
        
        # Initial Deal: 5 cards each
        for p in range(self.num_players):
            self.hands[p] = self.deck[:5]
            self.deck = self.deck[5:]
            
        self.face_up_card = self.deck.pop(0)
        self.face_up_suit = self.face_up_card // 8
        self.face_up_rank = self.face_up_card % 8
        
        self.current_player = (self.dealer + 1) % self.num_players
        self.phase = "BIDDING"
        self.bidding_round = 1
        self.passes_in_round = 0
        self.trump = None
        self.declarer = None
        
        # Tricks tracking
        self.tricks_played = 0
        self.current_trick = []  # List of (player_id, card)
        self.trick_history = []
        self.tricks_won_by_team = [0, 0] # Team 0 (P0, P2), Team 1 (P1, P3)
        self.raw_points_by_team = [0, 0]
        
        self.done = False

        # Forced Jack Exception
        if self.face_up_rank == 4: # Jack
            self.trump = self.face_up_suit
            self.declarer = self.current_player # TODO: If the face up is Jack, the player next to the dealer becomes the declarer
            self._finalize_bidding()
            
        return self._get_observation()

    def _finalize_bidding(self):
        """Called when a trump is chosen. Deals remaining cards."""
        self.phase = "PLAYING"
        # Deal remaining cards: Declarer gets face-up card + 2 more. Others get 3 more. TODO: If we are in bidding phase 1: the declarer takes the 
        # face up card. If in phase 2, no matter who chooses the trump the dealer takes the phase up card
        for p in range(self.num_players):
            if p == self.declarer:
                self.hands[p].append(self.face_up_card)
                self.hands[p].extend(self.deck[:2])
                self.deck = self.deck[2:]
            else:
                self.hands[p].extend(self.deck[:3])
                self.deck = self.deck[3:]
                
        # Declarer leads the first trick
        self.current_player = self.declarer

    def get_legal_actions(self):
        """Returns a boolean array of length 38 indicating legal actions for the current player."""
        legal = np.zeros(self.action_space_size, dtype=bool)
        
        if self.done:
            return legal
            
        if self.phase == "BIDDING":
            legal[32] = True # Pass is always an option in bidding TODO: if in bidding round 2 the dealer cannot pass, he must choose a suit but not the face up suit
            if self.bidding_round == 1:
                legal[33] = True # Accept
            elif self.bidding_round == 2:
                # Can choose any suit EXCEPT the face up suit
                for suit in range(4):
                    if suit != self.face_up_suit:
                        legal[34 + suit] = True
        
        elif self.phase == "PLAYING":
            hand = self.hands[self.current_player]
            
            # If leading the trick, can play any card TODO: not quite, if the declarer has yet to enter a trick with a trump, others cannot
            # enter with trump. In other word, the declarer is the first one to enter with a trump. Exception: If a player has only trump, he can play any trump
            if len(self.current_trick) == 0:
                for card in hand:
                    legal[card] = True
                return legal
                
            # Following trick logic
            led_card = self.current_trick[0][1] 
            led_suit = led_card // 8
            
            has_led_suit = any(c // 8 == led_suit for c in hand)
            has_trump = any(c // 8 == self.trump for c in hand)
            
            # Find the highest trump currently in the trick
            trick_trumps = [c[1] for c in self.current_trick if c[1] // 8 == self.trump]
            highest_trick_trump_val = max([self._get_card_value(t, is_trump=True)[1] for t in trick_trumps]) if trick_trumps else -1
            
            for card in hand:
                card_suit = card // 8
                card_val = self._get_card_value(card, is_trump=(card_suit == self.trump))[1]
                
                if has_led_suit: # TODO: Lead suit can be trump, so this would make all cards of the trump suit legal, without any respect for overruffing
                    if card_suit == led_suit:
                        legal[card] = True
                elif has_trump:
                    if card_suit == self.trump:
                        # Overruff rule: Must play a higher trump if possible
                        can_overruff = any(c // 8 == self.trump and self._get_card_value(c, True)[1] > highest_trick_trump_val for c in hand)
                        if can_overruff:
                            if card_val > highest_trick_trump_val:
                                legal[card] = True
                        else:
                            legal[card] = True # Have to play trump, but can't overruff
                else:
                    # Discard: No led suit, no trumps
                    legal[card] = True
                    
        return legal

    def step(self, action):
        if not self.get_legal_actions()[action]:
            raise ValueError(f"Illegal action {action} chosen by Player {self.current_player}")

        if self.phase == "BIDDING":
            self._handle_bidding_action(action)
        elif self.phase == "PLAYING":
            self._handle_playing_action(action)

        reward = [0, 0, 0, 0]
        if self.done:
            reward = self._calculate_final_rewards()
            # Rotate dealer for next game organically
            self.dealer = (self.dealer + 1) % self.num_players

        return self._get_observation(), reward, self.done, {}

    def _handle_bidding_action(self, action):
        if action == 32: # Pass
            self.passes_in_round += 1
            if self.passes_in_round == 4:
                if self.bidding_round == 1:
                    self.bidding_round = 2
                    self.passes_in_round = 0
                else:
                    # 4 passes in second round -> Redeal (done, 0 points) TODO: There can not be 4 passes in round 2, the dealer has to choose a suit
                    self.done = True
        elif action == 33: # Accept
            self.trump = self.face_up_suit
            self.declarer = self.current_player
            self._finalize_bidding()
        elif 34 <= action <= 37: # Choose Suit
            self.trump = action - 34
            self.declarer = self.current_player
            self._finalize_bidding()
            
        if self.phase == "BIDDING":
            self.current_player = (self.current_player + 1) % self.num_players

    def _handle_playing_action(self, action):
        card = action
        self.hands[self.current_player].remove(card)
        self.current_trick.append((self.current_player, card))
        
        if len(self.current_trick) < 4:
            self.current_player = (self.current_player + 1) % self.num_players
        else:
            # Evaluate trick
            winner, points = self._evaluate_trick()
            winning_team = winner % 2
            
            self.tricks_won_by_team[winning_team] += 1
            self.raw_points_by_team[winning_team] += points
            self.trick_history.append(self.current_trick)
            self.tricks_played += 1
            
            self.current_player = winner
            self.current_trick = []

            if self.tricks_played == 8:
                # Hand over, calculate Pasledu and Finish
                # Pasledu Rule: If winner of last trick is NOT the declarer, their team gets 10 raw points TODO: not quite,
                # The team who gets the last trick gets 10 points. The 'NOT the declarer' part is there because when we go from points to 'bile' we use the round rule for the team
                # who did not choose the trump, and calculate the bile for the winning team by doing 16 - the losing team's bile. 
                if winner != self.declarer:
                    self.raw_points_by_team[winning_team] += 10
                self.done = True

    def _evaluate_trick(self):
        led_suit = self.current_trick[0][1] // 8
        best_player = None
        best_rank_val = -1
        best_is_trump = False
        points = 0
        
        for player, card in self.current_trick:
            suit = card // 8
            is_trump = (suit == self.trump)
            pts, rank_val = self._get_card_value(card, is_trump)
            points += pts
            
            if not is_trump and not best_is_trump and suit == led_suit:
                if rank_val > best_rank_val:
                    best_rank_val = rank_val
                    best_player = player
            elif is_trump and best_is_trump:
                if rank_val > best_rank_val:
                    best_rank_val = rank_val
                    best_player = player
            elif is_trump and not best_is_trump:
                best_is_trump = True
                best_rank_val = rank_val
                best_player = player
                    
        return best_player, points

    def _get_card_value(self, card, is_trump):
        """Returns (Points, Internal Trick Power Rank)"""
        rank = card % 8
        # Ranks: 0=7, 1=8, 2=9, 3=10, 4=J, 5=Q, 6=K, 7=A
        if is_trump:
            points_map = {0:0, 1:0, 2:14, 3:10, 4:20, 5:3, 6:4, 7:11}
            # Trick hierarchy in trump: 7 < 8 < Q < K < 10 < A < 9 < J
            rank_map = {0:0, 1:1, 5:2, 6:3, 3:4, 7:5, 2:6, 4:7} 
        else:
            points_map = {0:0, 1:0, 2:0, 3:10, 4:2, 5:3, 6:4, 7:11}
            # Trick hierarchy non-trump: 7 < 8 < 9 < J < Q < K < 10 < A
            rank_map = {0:0, 1:1, 2:2, 4:3, 5:4, 6:5, 3:6, 7:7}
            
        return points_map[rank], rank_map[rank]

    def _calculate_final_rewards(self):
        """Calculates MARL team rewards utilizing the specific 'Bile' conversion."""
        if sum(self.tricks_won_by_team) == 0: # TODO: What is this? How can this be?
            return [0, 0, 0, 0] # Redeal edge case
            
        game_points = [0, 0]
        
        for team in range(2):
            if self.tricks_won_by_team[team] == 0:
                game_points[team] = -10 # Zero tricks penalty
            else:
                raw = self.raw_points_by_team[team]
                # Bile Rounding logic: ending in 5 rounds down (35->3), ending in 6 rounds up (66->7)
                remainder = raw % 10
                game_points[team] = (raw // 10) + (1 if remainder > 5 else 0)
                
        # Both partners share the exact same reward to ensure cooperative training
        return [
            game_points[0], # P0
            game_points[1], # P1
            game_points[0], # P2
            game_points[1]  # P3
        ]

    def _get_observation(self):
        """Returns the dictionary observation structure suited for PPO feature extraction."""
        return {
            "current_player": self.current_player,
            "hand": self.hands[self.current_player],
            "trump": self.trump,
            "trick_history": self.trick_history,
            "current_trick": self.current_trick
        }