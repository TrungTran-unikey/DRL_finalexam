from ursina import *
timer =21  
app=Ursina(borderless=False)
ground = Entity(model="cube",color=color.black,scale=20,texture="white_cube",texture_scale=20)
'''#image squence 
count_down = Animation('20-second/00', fps = 2  , scale = 5, Loop = False,autoplay=True)
def update():
    global timer
    timer -= time.dt
    if timer <= 0:
        timer = 0
        count_down.enabled = False
        if count_down.enabled == False:
            count_down_zero = Entity(model='quad', texture='20-second/00121.png', scale=5, position=(0,0,0))'''
EditorCamera()
app.run()