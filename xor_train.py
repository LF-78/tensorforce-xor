# MODEL
from xor_model import *  

# LOAD SAVED MODEL
if Path(savefile + '-10000.index').is_file():
    print('Loading saved model: '+savefile)
    learn_agent.load_model(savefile + '-10000')
else:
    print('Saved model not found: '+savefile)

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

# SAVE MODEL
print('Saving model: '+savefile)
learn_agent.save_model(savefile)

