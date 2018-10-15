import os
from tqdm import *
import base64
import time
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
slice_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
def prepare_tensor(path,save_path):
    img = Image.open(path)
    img = img.convert("RGB")
    img = np.array(img)
    actions = {
            'walk' : {
                'range': [(9,10),(10,11),(11,12)],
                'frames': [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8)]
                },
            'spellcast': {
                'range': [(1,2),(2,3),(3,4)],
                'frames': [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(6,7)]
                },
            'slash': {
                'range': [(14,15),(15,16),(16,17)],
                'frames':  [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(5,6),(5,6)]
                }
            }
    for action,params in actions.items():
        i = 0
        for row in params['range']:
            sprite = []
            for col in params['frames']:
                sprite.append(slice_transform(img[64*row[0]:64*row[1],64*col[0]:64*col[1],:]))
            os.makedirs(os.path.dirname(save_path+'/{}/{}.sprite'.format(action,i)),exist_ok=True)
            sprite_tensor = torch.stack(sprite)
            torch.save(sprite_tensor,save_path+'/{}/{}.sprite'.format(action,i))
            torchvision.utils.save_image(sprite_tensor,save_path+'/{}/{}.png'.format(action,i))
            i += 1


driver = webdriver.Firefox()
driver.get("http://gaurav.munjal.us/Universal-LPC-Spritesheet-Character-Generator/")
driver.maximize_window()

bodies = ['light','dark','dark2','darkelf','darkelf2','tanned','tanned2']
shirts = ['longsleeve_brown','longsleeve_teal','longsleeve_maroon','longsleeve_white']
hairstyles = ['green','blue','pink','raven','white','dark_blonde']
pants = ['magenta','red','teal','white','robe_skirt']
for body in tqdm(bodies):
    driver.execute_script("return arguments[0].click();",driver.find_element_by_id('body-'+body))
    time.sleep(0.5)
    for shirt in shirts:
        driver.execute_script("return arguments[0].click();",driver.find_element_by_id('clothes-'+shirt))
        time.sleep(0.5)
        for pant in pants:
            if pant=='robe_skirt':
                driver.execute_script("return arguments[0].click();",driver.find_element_by_id('legs-'+pant))
            else:
                driver.execute_script("return arguments[0].click();",driver.find_element_by_id('legs-pants_'+pant))
            time.sleep(0.5)
            for hair in hairstyles:
                driver.execute_script("return arguments[0].click();",driver.find_element_by_id('hair-plain_'+hair))
                time.sleep(0.5)
                name = body+"_"+shirt+"_"+pant+"_"+hair
                canvas = driver.find_element_by_id('spritesheet')
                canvas_base64 = driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);",canvas)
                canvas_png = base64.b64decode(canvas_base64)
                with open(str(name) + ".png","wb") as f:
                    f.write(canvas_png)
                save_path = './style-transfer/{}/{}/{}/{}'.format(body,shirt,pant,hair)
                slices = prepare_tensor(str(name) + ".png",save_path)
