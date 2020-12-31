import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
for file in os.listdir(BASE_DIR):
	if file.endswith('.png'):
		os.remove(file) 
		print('Remove Image{}'.format(file))
	else:
		pass