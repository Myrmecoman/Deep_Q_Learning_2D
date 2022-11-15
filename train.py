import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pyautogui
import cv2
import pathlib
import time
import os
import subprocess
import pytesseract


# https://keras.io/examples/rl/deep_q_network_breakout/
# https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
# to be evaluated : https://github.com/Z-T-WANG/ConvergentDQN


# env part -------------------------------------------------------------------------------------------------------------------------------
last_score = 0

def screen_and_process():
    # screenshoting
    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # croping and resizing
    score = image[0:50, 1820:1920]
    score = cv2.cvtColor(np.array(score), cv2.COLOR_BGR2GRAY)
    score = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #new_shape_score = [int(score.shape[1] / 2), int(score.shape[0] / 2)]
    #score = cv2.resize(score, new_shape_score)
    #print(new_shape_score)

    image = image[0:1080, 420:1500]
    new_shape_img = [84, 84]
    image = cv2.resize(image, new_shape_img)
    #print(new_shape_img)

    # saving
    #cv2.imwrite(current_dir + "/score.png", score)
    #cv2.imwrite(current_dir + "/image.png", image)

    # tesseract
    custom_config = r'--oem 3 --psm 6'
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    details = pytesseract.image_to_data(score, output_type=pytesseract.Output.DICT, config=custom_config, lang="eng")
    #print(details.keys())

    word = ""
    for s_word in details["text"]:
        if s_word != '':
            word = s_word
            break

    return image, int(word)


def reset():
    global last_score

    last_score = 0
    pyautogui.moveTo(1800, 1020)
    pyautogui.click()
    time.sleep(0.3)

    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = image[0:1080, 420:1500]
    new_shape_img = [84, 84]
    image = cv2.resize(image, new_shape_img)
    return image


def step(action, step_time = 0.03):
    global last_score
    #start_time = time.time()

    if (action == 0):
        pyautogui.press("Z")
        time.sleep(step_time)
    if (action == 1):
        pyautogui.press("S")
        time.sleep(step_time)
    if (action == 2):
        pyautogui.press("Q")
        time.sleep(step_time)
    if (action == 3):
        pyautogui.press("D")
        time.sleep(step_time)

    image, score = screen_and_process()

    adjusted = -1
    done = False
    if (score > last_score):
        done = True
        adjusted = 10
    if (score < last_score):
        adjusted = -15
    last_score = score

    #print(score)
    #print("%s sec" % (time.time() - start_time))
    return image, adjusted, done
# env part -------------------------------------------------------------------------------------------------------------------------------

# model part -------------------------------------------------------------------------------------------------------------------------------
num_actions = 4

def create_q_model():
    inputs = layers.Input(shape=(84, 84, 3,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)


# The first model makes the predictions for Q-values which are used to make an action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()

current_dir = str(pathlib.Path(__file__).parent.absolute())
if (not os.path.isfile(current_dir + "\\save\\checkpoint")):
    model.save_weights(current_dir + "\\save\\model_weights")
    model_target.save_weights(current_dir + "\\save\\model_target_weights")
model.load_weights(current_dir + "\\save\\model_weights")
model_target.load_weights(current_dir + "\\save\\model_target_weights")
# model part -------------------------------------------------------------------------------------------------------------------------------

# training part -------------------------------------------------------------------------------------------------------------------------------
# running game
process = subprocess.Popen(os.path.join(current_dir, "build\\DQN_2D.exe"))
time.sleep(5)


# In the Deepmind paper they use RMSProp however then Adam optimizer improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)

# Configuration paramaters for the whole setup
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
if (os.path.isfile(current_dir + "\\save\\arrays0.npz")):
    action_history = list(np.load(current_dir + "\\save\\arrays0.npz")["arr_0"])
    state_history = list(np.load(current_dir + "\\save\\arrays1.npz")["arr_0"])
    state_next_history = list(np.load(current_dir + "\\save\\arrays2.npz")["arr_0"])
    rewards_history = list(np.load(current_dir + "\\save\\arrays3.npz")["arr_0"])
    done_history = list(np.load(current_dir + "\\save\\arrays4.npz")["arr_0"])
    episode_reward_history = list(np.load(current_dir + "\\save\\arrays5.npz")["arr_0"])
    vals = list(np.load(current_dir + "\\save\\arrays6.npz")["arr_0"])
    running_reward = vals[0]
    episode_count = vals[1]
    frame_count = vals[2]
    print("Frame count : " + str(frame_count) + ", Episode count : " + str(episode_count) + ", Running reward: " + str(running_reward))

# Number of frames to take random action and observe output
epsilon_random_frames = 1000
# Number of frames for exploration
epsilon_greedy_frames = 10000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 500
# Using huber loss for stability
loss_function = keras.losses.Huber()

while True:  # Run until solved
    state = np.array(reset())
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values from environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done = step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            template = template.format(running_reward, episode_count, frame_count)
            print(template)
            file = open(current_dir + "\\training.txt", "a")  # append mode
            file.write(template + "\n")
            file.close()

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    model.save_weights(current_dir + "\\save\\model_weights")
    model_target.save_weights(current_dir + "\\save\\model_target_weights")
    other_nbs = [running_reward, episode_count, frame_count]
    np.savez(current_dir + "\\save\\arrays0", action_history)
    np.savez(current_dir + "\\save\\arrays1", state_history)
    np.savez(current_dir + "\\save\\arrays2", state_next_history)
    np.savez(current_dir + "\\save\\arrays3", rewards_history)
    np.savez(current_dir + "\\save\\arrays4", done_history)
    np.savez(current_dir + "\\save\\arrays5", episode_reward_history)
    np.savez(current_dir + "\\save\\arrays6", other_nbs)

    if running_reward > 40:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
# training part -------------------------------------------------------------------------------------------------------------------------------

# kill the process
process.kill()
