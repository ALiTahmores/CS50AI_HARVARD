import math
import random
import time


class Nim:
    def __init__(self, initial=[1, 3, 5, 7]):
        """
        Initialize the game board with given piles.
        Each board has:
        - `piles`: list representing the number of objects in each pile.
        - `player`: 0 or 1 indicating whose turn it is.
        - `winner`: None, 0, or 1 indicating the winner.
        """
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """
        Returns a set of all possible actions (i, j) for a given `piles` state.
        Action (i, j) represents removing `j` items from pile `i`.
        """
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        """
        Returns the other player.
        """
        return 0 if player == 1 else 1

    def switch_player(self):
        """
        Switch the current player to the other player.
        """
        self.player = Nim.other_player(self.player)

    def move(self, action):
        """
        Executes the given `action` for the current player.
        """
        pile, count = action

        # Validate move
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        # Perform action and update game state
        self.piles[pile] -= count
        self.switch_player()

        # Check if game has a winner
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI:
    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with:
        - `q`: Q-learning dictionary.
        - `alpha`: Learning rate.
        - `epsilon`: Exploration factor.
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Update the Q-value for a given state-action pair.
        """
        old_q = self.get_q_value(old_state, action)
        future_rewards = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old_q, reward, future_rewards)

    def get_q_value(self, state, action):
        """
        Retrieve the Q-value for a state-action pair.
        Defaults to 0 if not set.
        """
        key = (tuple(state), tuple(action))
        return self.q.get(key, 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value using the Q-learning formula.
        """
        key = (tuple(state), tuple(action))
        self.q[key] = old_q + self.alpha * (reward + future_rewards - old_q)

    def best_future_reward(self, state):
        """
        Find the maximum Q-value for all possible actions in a given state.
        """
        actions = Nim.available_actions(state)
        if not actions:
            return 0
        return max(self.get_q_value(state, action) for action in actions)

    def choose_action(self, state, epsilon=True):
        """
        Choose an action using epsilon-greedy strategy.
        """
        actions = Nim.available_actions(state)
        if not actions:
            raise Exception("No available actions")

        if epsilon and random.random() < self.epsilon:
            return random.choice(list(actions))
        else:
            return max(actions, key=lambda action: self.get_q_value(state, action))


def train(n):
    """
    Train the AI by simulating `n` games.
    """
    ai = NimAI()

    for i in range(n):
        print(f"Training game {i + 1}/{n}")
        game = Nim()
        last_moves = {0: {"state": None, "action": None}, 1: {"state": None, "action": None}}

        while True:
            state = game.piles.copy()
            action = ai.choose_action(state)

            # Record the current move
            last_moves[game.player] = {"state": state, "action": action}

            # Make the move and transition to the new state
            game.move(action)
            new_state = game.piles.copy()

            # Check if the game has ended
            if game.winner is not None:
                # Update Q-values for the winning and losing moves
                ai.update(state, action, new_state, -1)
                if last_moves[game.player]["state"]:
                    ai.update(
                        last_moves[game.player]["state"],
                        last_moves[game.player]["action"],
                        new_state,
                        1
                    )
                break

            # Update Q-values for ongoing moves
            if last_moves[game.player]["state"]:
                ai.update(
                    last_moves[game.player]["state"],
                    last_moves[game.player]["action"],
                    new_state,
                    0
                )

    print("Training complete.")
    return ai


def play(ai, human_player=None):
    """
    Play a game of Nim against the AI.
    """
    if human_player is None:
        human_player = random.randint(0, 1)

    game = Nim()

    while True:
        print("\nPiles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        if game.player == human_player:
            print("Your Turn")
            while True:
                try:
                    pile = int(input("Choose Pile: "))
                    count = int(input("Choose Count: "))
                    if (pile, count) in available_actions:
                        break
                except ValueError:
                    pass
                print("Invalid move, try again.")
        else:
            print("AI's Turn")
            pile, count = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {count} from pile {pile}.")

        game.move((pile, count))

        if game.winner is not None:
            print("\nGAME OVER")
            print("Winner:", "Human" if game.winner == human_player else "AI")
            return
