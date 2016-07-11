#  coding: utf-8
# 用PIL导入，压缩，再写回jpg

from PIL import Image

nClass = 13
label_names = ['glass','city','earth','plane','dog','flowers','gold','car','gun','worldcup','fruits','sky','fireworks']
label_names = {k:v  for k,v in enumerate( label_names )}

dim_val = 34

dim_vals = dim_val * dim_val * 3

for label in range( nClass ):
	file_name_ = ('./%d/' % label) + label_names[ label ] + '_'
	
	for pic_No in range(200):
		file_name = file_name_ + '%04d.jpg' % pic_No
		try:
			in_img = Image.open( file_name )
		except:
			break
		in_img = in_img.resize( (dim_val, dim_val), Image.ANTIALIAS )
		in_img.save( file_name )