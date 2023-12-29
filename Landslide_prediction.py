‘‘‘
模型构建
’’’
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from numpy import column_stack
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
from keras.callbacks import ModelCheckpoint

#读取 降雨致灾滑坡样本数据.csv
dataset = read_csv('降雨致灾滑坡样本数据.csv',header=0,index_col=0)

#标准化
ss = StandardScaler()
values = dataset.values
print('shape', values.shape)
values = ss.fit_transform(values)
values = values.astype('float32')

#提出预测数据集
dataset_z = values[-1:,:-1]

#划分影响因素与滑距
values = values[:-1,:]
dataset_x = values[:,0:-1]
dataset_y = values[:,-1]
#目标标签化
mean = dataset_y.mean(axis=0)
dataset_y -= mean
std = dataset_y.std(axis=0)
dataset_y /= std
#划分训练集、训练目标、测试集、测试目标
from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target=train_test_split(dataset_x,dataset_y,test_size=0.25)
print('train_data', train_data,
	'train_target', train_target)


#建立模型与优化
adam = optimizers.Nadam(lr=0.001, 
beta_1=0.9,
beta_2=0.999, 
epsilon=1e-08, 
schedule_decay=0.000005)
checkpointer = ModelCheckpoint(filepath="best_model1.h5", 
verbose=1, 
save_best_only=True)
#最终模型
def build_model():
	model = models.Sequential()
	model.add(layers.Dense(64, activation = 'relu', 
input_shape = (train_data.shape[1],)))
	#model.add(layers.Dropout(0.5))
	#model.add(layers.Dense(256, activation = 'relu'))
	#model.add(layers.Dropout(0.5))
	model.add(layers.Dense(128, activation = 'relu'))	
	model.add(layers.Dense(32, activation = 'relu'))
	#model.add(layers.Dropout(0.5))
	model.add(layers.Dense(1))
	model.compile(optimizer='RMSprop', 
loss='mse', 
metrics=['mae'])
	return model
	
#K折交叉验证
k = 4
num_val_samples = len(train_data) //k
num_epochs = 10000
all_scores = []

for i in range(k):
	print(i)
	val_data = train_data[i * num_val_samples: (i + i) * num_val_samples]
	val_target = train_target[i * num_val_samples: (i + 1) * num_val_samples]
	partial_train_data = np.concatenate(
		[train_data[:i * num_val_samples],
		 train_data[(i + 1) * num_val_samples:]],axis=0)
	partial_train_targets = np.concatenate(
		[train_target[:i * num_val_samples],
		 train_target[(i + 1) * num_val_samples:]],axis=0)

#模型训练
model = build_model()

'''
#loss与val_loss曲线
history = model.fit(partial_train_data, partial_train_targets, 
epochs=num_epochs, 
batch_size=4,
validation_data=(test_data, test_target),
verbose=0, 
shuffle=False,
callbacks=[checkpointer])

#Chart of training and validation loss
#model = models.load_model('best_model.h5')
loss = history.history['loss']
val_loss = history.history['val_loss']


def smooth_curve(points, factor = 0.9):
	smoothed_points = []
	for point in points:
		if smoothed_points:
			previous = smoothed_points[-1]
			smoothed_points.append(previous * factor + point * (1 - factor))
		else:
			smoothed_points.append(point)
	return smoothed_points
smooth_loss = smooth_curve(loss[100:])
smooth_val_loss = smooth_curve(val_loss[100:])

plt.plot(range(1, len(smooth_loss) + 1), smooth_loss, 'bo', label = 'loss')
plt.plot(range(1, len(smooth_val_loss) + 1), smooth_val_loss, 'b', label = 'val_loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
'''

# 模型预测
model = models.load_model('最优模型.h5')
yhat = model.predict(test_data)
print('yhat',yhat)
#测试输出反标签化
yhat *= std
yhat += mean
#测试输出反归一化
inv_yhat = column_stack((test_data,yhat))
inv_yhat = ss.inverse_transform(inv_yhat)
print(inv_yhat)
inv_yhat = inv_yhat[:,-1]
print('predict', inv_yhat)

#预测输出
zhat = model.predict(dataset_z)
print('zhat', zhat)
#预测反标签化
zhat *= std
zhat += mean
print('zhat1',zhat)
#预测输出反归一化
inv_zhat = column_stack((dataset_z,zhat))
inv_zhat = ss.inverse_transform(inv_zhat)
print(inv_zhat)
inv_zhat = inv_zhat[:,-1]
print('predict_z', inv_zhat)

#测试真实值反归一化
inv_y = column_stack((test_data, test_target))
inv_y = ss.inverse_transform(inv_y)
print('inv_y', inv_y )
inv_y = inv_y[:,-1]

#画出对比折线图
plt.figure()
plt.plot(range(len(inv_yhat)),inv_yhat, 'b', label="predict")
plt.plot(range(len(inv_y)),inv_y, 'r', label="test")
plt.legend(loc="upper right")
plt.xlabel("avalanche number")
plt.ylabel('travel distance')
plt.show()

#模型性能确定
eval_results = model.evaluate(test_data, test_target)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))

import tkinter as tk
from tkinter import messagebox
from keras import models
import pandas as pd
from pandas import read_csv
import numpy as np
from pandas import DataFrame
from numpy import column_stack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
#建立窗口
window = tk.Tk()
window.title('滑坡预测')
window.geometry('450x300')
window.iconphoto(False, tk.PhotoImage(file='小标3.png'))

#影响因素部分
factors = tk.LabelFrame(window,
		        width=200, 
		        height=260,
		        text='影响因素', 
		        padx=20,
		        pady=0)
factors.place(x=20,y=20)

#预测结果部分
Prediction = tk.LabelFrame(window,
	     	             width=190, 
	   	             height=135,
		             text='预测距离',
                                               padx=15,
		             pady=15)
Prediction.place(x=240,y=20)

#指令执行部分
switch = tk.LabelFrame(window,
		       width = 190,
		       height=115,
		       text = "", 
		       padx = 13, 
		       pady = 19 )
switch.place(x = 240, y = 165)

#影响因素与预测值分别放置
tk.Label(factors, text='坡体体积(×10⁴m³)',font=('微软雅黑',10)).place(x=0, y=0)
tk.Label(factors, text='最大高差(m)',font=('微软雅黑',10)).place(x=0, y=45)
tk.Label(factors, text='滑源区坡度(°)',font=('微软雅黑',10)).place(x=0, y=90)
tk.Label(factors, text='地层倾角(°)',font=('微软雅黑',10)).place(x=0, y=135)
tk.Label(factors, text='坡脚坡度(°)',font=('微软雅黑',10)).place(x=0, y=180)
tk.Label(Prediction, text='最大水平滑距(m)',font=('微软雅黑',12)).place(x=0, y=0)

#坡体体积
var_tj = tk.StringVar()
var_tj.set('>0')
entry_tj = tk.Entry(factors,textvariable=var_tj)
entry_tj.place(x=0,y=25)
#最大高差
var_gc = tk.StringVar()
var_gc.set('100～1000')
entry_gc = tk.Entry(factors,textvariable=var_gc)
entry_gc.place(x=0,y=70)
#滑源区坡度
var_pd = tk.StringVar()
var_pd.set('0～90')
entry_pd = tk.Entry(factors,textvariable=var_pd)
entry_pd.place(x=0,y=115)
#地层倾角
var_qj = tk.StringVar()
var_qj.set('0～180')
entry_qj = tk.Entry(factors, textvariable=var_qj)
entry_qj.place(x=0,y=160)
#坡脚坡度
var_pj = tk.StringVar()
var_pj.set('0～90')
entry_pj = tk.Entry(factors, textvariable=var_pj)
entry_pj.place(x=0,y=205)
#最大水平滑距预测
var_pre_jl=tk.StringVar()
#entry_pre_jl = tk.Entry(Prediction, width=20,
  		       textvariable = var_pre_jl,
		       state='disabled')
entry_pre_jl = tk.Entry(Prediction,  width=20,
		     textvariable = var_pre_jl)
entry_pre_jl.place(x=0,y=45)

#定义“预测”命令
def var_pre():
    entry_pre_jl.delete(0, 'end')
    tj = var_tj.get()
    gc = var_gc.get()
    pd = var_pd.get()
    qj = var_qj.get()
    pj = var_pj.get()
    #输入数字判断
    #判断是否为数字
    try:
        val_1=float(tj)
        val_1 *= 10000
        val_2=float(gc)
        val_3=float(pd)
        val_4=float(qj)
        val_5=float(pj)
    except:
        messagebox.showwarning('警告', '请输入数字！') 
    print(val_1)
    #判断数据大小范围：
    if val_1 > 0:
        pass
    else:
        messagebox.showwarning('警告', '输入值超出范围!')
        entry_tj.delete(0, 'end')
        sys.exit(0)     
    if val_2 > 0:
        pass
    else:
        messagebox.showwarning('警告', '输入值超出范围!')
        entry_gc.delete(0, 'end')
        sys.exit(0)
    if val_3 > 0 and val_3 < 90:
        pass
    else:
        messagebox.showwarning('警告', '输入值超出范围!')
        entry_pd.delete(0, 'end')
        sys.exit(0)
    if val_4 >= 0 and val_4 < 180:
        pass
    else:
        messagebox.showwarning('警告', '输入值超出范围!')
        entry_qj.delete(0, 'end')
        sys.exit(0)
    if val_5 >= 0 and val_5 <= 90:
        pass
    else:
        messagebox.showwarning('警告', '输入值超出范围!')
        entry_pj.delete(0, 'end')
        sys.exit(0)
    if val_2 > 100 and val_2 <1000:
        pass
    else:
        messagebox.showinfo('提示', '高差在100～1000m之外的滑坡滑距预测可能不精确！')
        print(val_2)
    #字典转Dataframe:
    values_p = {'tj':[val_1],
                'gc':[val_2],
                'pd':[val_3],
                'qj':[val_4],
                'pj':[val_5]}
    import  pandas as pd
    from io import StringIO
    df = pd.DataFrame(data = values_p)
    df_string = StringIO(df.to_csv()) 
    del df
    values_p = pd.read_csv(df_string,header=0, index_col=0)
    values_p = values_p.astype('float32')
    #标准化：
    dataset = read_csv('4_12.csv', header=0, index_col=0)
    values = dataset.values
    values = values.astype('float32')
    dataset_a = values[:,:-1]
    mean1 = dataset_a.mean(axis=0)
    values_p -= mean1
    std1 = dataset_a.std(axis=0)
    values_p /= std1
    print('values_p',values_p)
    
    values = values[:-1, :]
    mean2 =values.mean(axis=0)
    std2 = values.std(axis=0)
    dataset_x = values[:, :-1]
    dataset_y = values[:, -1]
    #标签化中间值:
    mean = dataset_y.mean(axis=0)
    dataset_y -= mean
    std =dataset_y.std(axis=0)
    dataset_y /= std
    mean3 = dataset_y.mean(axis=0)
    std3 = dataset_y.std(axis=0)
    #反标准化中间值：
    model = models.load_model('best_model1.h5')
    phat = model.predict(values_p)
    print('phat',phat)
    phat *= std3
    phat += mean3
    inv_phat = column_stack((values_p, phat))
    print('inv_phat',inv_phat)
    inv_phat *= std2
    inv_phat += mean2
    print(inv_phat)
    inv_phat = inv_phat[:, -1]
    print('predict', inv_phat)
    if inv_phat > 0:
        entry_pre_jl.insert(0, int(inv_phat))
    else:
        messagebox.showerror('错误', '对不起，无法预测该滑坡滑距！')
        sys.exit(0)
#定义“图例”命令
image_file = tk.PhotoImage(file='图例.png')
def var_pic():
    window_var_pic = tk.Toplevel(window)
    window_var_pic.geometry('600x300')
    window_var_pic.title('图例')
    window_var_pic.iconphoto(False, tk.PhotoImage(file='小标3.png'))
    canvas = tk.Canvas(window_var_pic, height=300, 
		    width=600,
		    bg='white')
    image = canvas.create_image(20, 20, anchor='nw', 
			     image=image_file)
    canvas.pack()
    btn_pic.config(state = 'disabled')
#定义“刷新”命令
def var_clr():
    if messagebox.askyesno('提示','请问是否刷新？'):
        entry_gc.delete(0, 'end')
        entry_tj.delete(0, 'end')
        entry_pd.delete(0, 'end')
        entry_qj.delete(0, 'end')
        entry_pj.delete(0, 'end')
        entry_pre_jl.delete(0, 'end')
    else:
        print('no')
#定义提示：
def point_out():
    window_var_point_out = tk.Toplevel(window)
    window_var_point_out.geometry('300x100')
    window_var_point_out.title('注意事项')
    window_var_point_out.iconphoto(False, tk.PhotoImage(file='小标3.png'))
    tk.Label(window_var_point_out,
             text='本软件适用于降雨致灾滑坡，\n坡体高差会对滑距预测造成较大影响，\n对于高差在100～1000m内预测较准确。',
             font=('微软雅黑',12),
             padx=10,pady=15).place(x=0, y=0)
def on_closing():
    if messagebox.askokcancel("提示", "请问是否关闭？"):
        window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)

#执行区按钮设置
btn_calc = tk.Button(switch, text='预测',
		  width = 9, 
		  height = 1, 
		  command = var_pre,).place(x=0,y=0)
btn_pic = tk.Button(switch,text='图例', 
		width = 9, 
		height = 1,
		state='normal', 
		command=var_pic).place(x=85,y=0)
btn_clear = tk.Button(switch,text='刷新',
		   width = 9,
		   height = 1,
		   command = var_clr).place(x=0,y=45)
btn_quit = tk.Button(switch,text='提示',
		  width = 9,
		  height = 1,
		  command=point_out).place(x=85,y=45)

window.mainloop()
