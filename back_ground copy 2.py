from ursina import * 

from env import *
from PPO import *

app = Ursina(borderless=False)
counter = 0
timer = 21
countdown_activate = True
iterations = 0
LOAD = False
TRAIN = True

screen = Animation('20-second/00', 
                       fps = 2  ,
                         position= (6,6,8.5),
                        scale= (7,4,0),
                           Loop = False,
                           autoplay=True)

logo = Entity(model="quad",
              texture="logo.png",
              position = (0,4,8.5),
              scale = (8,4,8))

ground = Entity(model="plane",
                position=(0,0,0),
                scale=20,
                color = color.gray,
                texture='white_cube',
                texture_scale=(20,20,20),
                collider='box')
wall_right = Entity(model="cube",
              color=color.gray,
              position=(9.5,1.5,0),
              texture='white_cube',
              scale=(1,3,18),
              texture_scale=(2,2,20),
              collider='box')
wall_left = duplicate(wall_right, x=-9.5)
wall_after = Entity(model="cube",
              position=(0,4,9.5),
              scale=(20,8,1),
              collider='box',
              color = color.white)
wall_before = Entity(model="cube",
             color=color.clear,
              position=(0,0.75,-9.5),
              scale=(20,1.5,1),
              collider='box')
Chaser = Entity(model="cube",
                color=color.red,
                position=(0,0.5,0),
                collider='box',
                scale=.5)
Evader = Entity(model="cube",
                color=color.blue, 
                position=(0,0.5,2),
                collider='box',
                scale=.5)

def update():
    global iterations
    global counter
    global timer
    global countdown_activate
    global LOAD
    
    counter += time.dt

    # if countdown_activate:
    env.time -= time.dt
    

    try:
        if LOAD:
            Chaser_agent.load_model()
            Evader_agent.load_model()
            LOAD = False

            if TRAIN:
                eps = 1
            else:
                eps = 0

            # Chaser_agent.eps = eps
            # Evader_agent.eps = eps
        
    except:
        pass

    # Get the current state of the environment
    state = env.state

    # Use the agents to determine the actions
    Chaser_action, Chaser_probs, Chaser_values = Chaser_agent.choose_action(state['Chaser'])
    Evader_action, Evader_probs, Evader_values = Evader_agent.choose_action(state['Evader'])

    action = {
        'Chaser': Chaser_action,
        'Evader': Evader_action,
    }

    next_state, reward, done, _ = env.step(action)

    # Let the agents remember the state, action, reward, next state, and done
    Chaser_agent.remember(state['Chaser'], Chaser_action, Chaser_probs, Chaser_values, reward['Chaser'], done)
    Evader_agent.remember(state['Chaser'], Evader_action, Evader_probs, Evader_values, reward['Evader'], done)
        
    if counter >= 0.01:
        counter = 0

        # Let the agents learn from the memory
        Chaser_agent.learn()
        Evader_agent.learn()

    if env.done:
        print('-'*50)
        print(f'#ITERATION: {iterations}')
        print(f'Chaser reward: {reward["Chaser"]}, Evader reward: {reward["Evader"]}') #, Epsilon: {Chaser_agent.eps}')
        # print(f'Chaser highest Q-value\' action: {Chaser_agent.model.forward(T.tensor(state["Chaser"], dtype=T.float).to(Chaser_agent.device)).argmax().item()}, Q-value: {Chaser_agent.model.forward(T.tensor(state["Chaser"], dtype=T.float).to(Chaser_agent.device)).max().item()}')
        # print(f'Evader highest Q-value\' action: {Evader_agent.model.forward(T.tensor(state["Evader"], dtype=T.float).to(Evader_agent.device)).argmax().item()}, Q-value: {Evader_agent.model.forward(T.tensor(state["Evader"], dtype=T.float).to(Evader_agent.device)).max().item()}')
        iterations += 1
        # save the models
        Chaser_agent.save_models()
        Evader_agent.save_models()

    if env.time <= 0:
        timer = '0'
        screen.enabled = False
        if screen.enabled == False:
            count_down_zero = Entity(model='quad', 
                                     texture='20-second/00121.png', 
                                     position= (6,6,8.5),
                                    scale= (7,4,0))
            destroy(screen)
            try:
                destroy(screen_chaser)
                destroy(screen_evader)
            except:
                pass
            screen_evader= Entity(model="quad",
                                texture="evader.png",
                                position=(6,6,8.5),
                                scale=(7,4,0))
            env.reset()

    if env.Chaser.intersects(env.Evader):
        
        destroy(screen)
        try:
            destroy(screen_chaser)
            destroy(screen_evader)
        except:
            pass
        screen_chaser= Entity(model="quad",
                                texture="chaser.png",
                                position=(6,6,8.5),
                                scale=(7,4,0))
        env.reset()

if __name__ == '__main__':
    env = ChaserEvader(Chaser, Evader)
    env.reset()
    n_actions = env.action_space.n
    input_dims = env.observation_space.shape
    Chaser_agent = Agent(n_actions=n_actions, input_dims=input_dims)
    Evader_agent = Agent(n_actions=n_actions, input_dims=input_dims)
    Sky()
    EditorCamera()
    camera.position = (0,5,0)
    
    app.run()