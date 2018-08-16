from flask import Flask
import pickle
import tensorflow as tf
from xgboost.sklearn import XGBRegressor
from sklearn import linear_model
import numpy as np
from flask import request
from flask import render_template
from model import my_model

app = Flask(__name__,template_folder='templates')


with tf.Session() as sess:
	saver = tf.train.import_meta_graph("model.ckpt.meta")
	saver.restore(sess,'model.ckpt')
	

graph = tf.get_default_graph()
pred = graph.get_tensor_by_name('dense_4/Relu:0')
X_t = graph.get_tensor_by_name('Placeholder:0')



filehandler = open('model.pickle', 'rb')
model = pickle.load(filehandler)
print(model.__module__)
filehandler.close()

@app.route("/")
def predict():
	return render_template('index.html')
 
@app.route('/house')
def house_price():
    # show the user profile for that user
	full_sq  = request.args.get('full_sq')
	life_sq  = request.args.get('life_sq')
	num_rooms= request.args.get('num_rooms')
	sub_area = request.args.get('sub_area')
	#print(full_sq,life_sq,num_rooms,sub_area)
	
	arr = np.array([full_sq,life_sq,num_rooms,sub_area]).reshape(1,-1)
	X_nn = model.predict(arr)
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph("model.ckpt.meta")
		saver.restore(sess,'model.ckpt')
		#saver.restore(sess, "C:\\Users\\USER\\AnacondaProjects\\.ipynb_checkpoints\\model.ckpt.meta")
	
		[pred_val] = sess.run([pred],feed_dict={X_t:X_nn})
	print(pred_val[0,0])
	return render_template('predict.html', price=pred_val[0,0])
if __name__ == "__main__":
	app.run()