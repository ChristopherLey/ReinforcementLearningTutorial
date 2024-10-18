import argparse
import gymnasium as gym
import numpy as np
from Agent import Agent
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training",
        "-t",
        dest="training",
        help="training mode",
        action="store_true",
    )
    parser.add_argument(
        "--environment",
        "-e",
        dest="environment",
        help="set the environment",
        default="LunarLander-v2",
    )
    args = parser.parse_args()
    horizon = 50
    batch_size = 5
    n_epochs = 4
    alpha = 0.001
    n_games = 400
    average_span = n_games // 10
    if args.training:
        env = gym.make(args.environment, max_episode_steps=200)

        agent = Agent(
            n_actions=env.action_space.n,
            input_dims=env.observation_space.shape,
            alpha=alpha,
            gamma=0.99,
            n_epochs=n_epochs,
            batch_size=batch_size,
            horizon=horizon
        )

        best_score = env.reward_range[0]
        score_history = []

        learn_iters = 0
        n_steps = 0

        for i in range(n_games):
            observation, info = env.reset()
            is_terminated = False
            score = 0
            while not is_terminated:
                action, log_probability, value = agent.choose_action(observation)
                observation_, reward, is_terminated, truncated, info = env.step(action)
                n_steps += 1
                score += reward
                agent.remember(observation, action, log_probability, value, reward, is_terminated)
                if n_steps % horizon == 0:
                    agent.learn()
                    learn_iters += 1
                observation = observation_
            score_history.append(score)
            if len(score_history) > average_span:
                avg_score = np.mean(score_history[-average_span:])

                if avg_score > best_score:
                    best_score = avg_score
                    agent.save_models()

                print(f"episode {i}, score {score:.1f}, avg_score {avg_score:.1f}, learn_iters {learn_iters}")
            else:
                print(f"episode {i}, score {score:.1f}, learn_iters {learn_iters}")
        env.close()
        x = [i+1 for i in range(len(score_history))]
        filename = f"{args.environment}.png"
        plt.plot(x, score_history)
        plt.title(args.environment)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.savefig(filename)
        plt.show()

    else:
        env = gym.make(args.environment, render_mode="human")

        agent = Agent(
            n_actions=env.action_space.n,
            input_dims=env.observation_space.shape,
            alpha=alpha,
            gamma=0.99,
            n_epochs=n_epochs,
            batch_size=batch_size,
            horizon=horizon
        )
        agent.load_models()
        observation, info = env.reset()
        score = 0
        for _ in range(1000):
            action, log_probability, value = agent.choose_action(observation)
            observation_, reward, is_terminated, truncated, info = env.step(action)
            score += reward
            if is_terminated or truncated:
                env.reset()

        print(f"score: {score}")
        env.close()


if __name__ == "__main__":
    main()


