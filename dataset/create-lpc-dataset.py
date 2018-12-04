import base64
import time
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
slice_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
def prepare_tensor(path):
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
    slices = [] 
    for action,params in actions.items():
        for row in params['range']:
            sprite = []
            for col in params['frames']:
                sprite.append(slice_transform(img[64*row[0]:64*row[1],64*col[0]:64*col[1],:]))
            slices.append(torch.stack(sprite))
    return slices 

driver = webdriver.Firefox()
driver.get("http://gaurav.munjal.us/Universal-LPC-Spritesheet-Character-Generator/")
driver.maximize_window()

bodies = ['light','dark','dark2','darkelf','darkelf2','tanned','tanned2']
shirts = ['longsleeve_brown','longsleeve_teal','longsleeve_maroon','longsleeve_white']
hairstyles = ['green','blue','pink','raven','white','dark_blonde']
pants = ['magenta','red','teal','white','robe_skirt']
train = 0
test = 0
for id_body, body in enumerate(bodies):
    driver.execute_script("return arguments[0].click();",driver.find_element_by_id('body-'+body))
    time.sleep(0.5)
    for id_shirt, shirt in enumerate(shirts):
        driver.execute_script("return arguments[0].click();",driver.find_element_by_id('clothes-'+shirt))
        time.sleep(0.5)
        for id_pant, pant in enumerate(pants):
            if pant=='robe_skirt':
                driver.execute_script("return arguments[0].click();",driver.find_element_by_id('legs-'+pant))
            else:
                driver.execute_script("return arguments[0].click();",driver.find_element_by_id('legs-pants_'+pant))
            time.sleep(0.5)
            for id_hair, hair in enumerate(hairstyles):
                driver.execute_script("return arguments[0].click();",driver.find_element_by_id('hair-plain_'+hair))
                time.sleep(0.5)
                name = body+"_"+shirt+"_"+pant+"_"+hair
                print("Creating character: "  + "'" + name)
                canvas = driver.find_element_by_id('spritesheet')
                canvas_base64 = driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);",canvas)
                canvas_png = base64.b64decode(canvas_base64)
                with open(str(name) + ".png","wb") as f:
                    f.write(canvas_png)
                slices = prepare_tensor(str(name) + ".png")
                p = torch.rand(1).item() <= 0.1 #Randomly add 10% of the characters created in the test set
                for id_action, sprites in enumerate(slices):
                    if p is True:
                        test += 1
                        path = './lpc-dataset/test/%d.sprite' % test
                    else:
                        train += 1
                        path = './lpc-dataset/train/%d.sprite' % train

                    final_sprite = {
                            'body': id_body,
                            'shirt': id_shirt,
                            'pant': id_pant,
                            'hair': id_hair,
                            'action': id_action // 3,
                            'sprite': sprites
                    }
                    print(id_body,id_shirt,id_pant,id_hair,id_action // 3)
                    torch.save(final_sprite, path)
 
print("Dataset is Ready.Training Set Size : %d. Test Set Size : %d " % (train,test))
