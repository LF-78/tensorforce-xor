# MODEL
from xor_model import *  

# LOAD SAVED MODEL
if Path(savefile + '.index').is_file():
    print('Loading saved model: '+savefile)
    learn_agent.load_model(savefile)
else:
    print('Saved model not found: '+savefile)

print('Training for: ' + str(training_length) + ' episodes')

# STATS
max_reward = float('-Inf')
min_reward = float('Inf')
tot_reward = 0

# TRAINING LOOP
for j in range(training_length):
    
    # NEW EPISODE
    step_reward = 0
    episode_reward = 0
    terminated = False
    x1 = 0
    x2 = 0
    y = 0
    learn_agent.reset()

    # EPISODE STEPS    
    for i in range(episode_size):
    
        # UPDATE STATE
        x1 = np.random.randint(0,2)
        x2 = np.random.randint(0,2)

        # ACTIONS
        y = learn_agent.act(state=[x1,x2], deterministic=False)
        
        # REWARD
        if y == (x1 ^ x2):
            step_reward = 1
        else:
            step_reward = 0
        if i == episode_size-1:
            terminated = True
        else:
            terminated = False
        learn_agent.observe(reward=step_reward, terminal=terminated)
        episode_reward = episode_reward + step_reward

    print('Episode: ' + str(j) + ' / Reward: ' + str(episode_reward))
    max_reward = max(max_reward, episode_reward)
    min_reward = min(min_reward, episode_reward)
    tot_reward = tot_reward + episode_reward

    # SAVE MODEL & STATS
    if (j+1) % save_frequency == 0:
        print('Episodes: ' + str(j+1) + ' / AVG Reward: ' + str(tot_reward/save_frequency) + ' / MAX Reward: ' + str(max_reward) + ' / MIN Reward: ' + str(min_reward) + ' / Saving model: '+savefile)
        learn_agent.model.save_model(savefile, False)
        max_reward = float('-Inf')
        min_reward = float('Inf')
        tot_reward = 0

