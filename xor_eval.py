# MODEL
from xor_model import *  

# LOAD SAVED MODEL
if Path(savefile + '.index').is_file():
    print('Loading saved model: '+savefile)
    learn_agent.load_model(savefile)
    
    # EVALUATE LOOP
    while True:
        x1 = input('X1: ')
        x2 = input('X2: ')
        y = learn_agent.act(state=[x1,x2], deterministic=True)
        print('Y: ' + str(y))
        print('--')

else:
    print('Saved model not found: '+savefile)

