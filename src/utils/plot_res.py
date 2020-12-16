import matplotlib.pyplot as plt
import pickle
import numpy as np

def plot(logdata, modelname="Naïve Model", printFlag=False, savepath=None):
    lst_iter = list(range(len(logdata['train_loss'])))
    plt.figure(figsize=(10,10))
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    ax1.plot(logdata['train_loss'], '-b', label='loss')
    ax1.set_xlabel("n iteration")
    ax1.legend(loc='upper right')
    ax1.set_title(f"Loss of the {modelname}")
    ax2.plot(lst_iter, logdata['train_acc'], '-r', label='train accuracy')
    ax2.plot(lst_iter, logdata['val_acc'], '-g', label='test accuracy')
    ax2.set_xlabel("n iteration")
    ax2.legend(loc='upper left')
    ax2.set_title(f"Accuracy of the {modelname}")
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)        
    acc = logdata['train_acc']
    val_acc = logdata['val_acc']
    loss = logdata['train_loss']                 
    lid = np.argmin(loss)
    aid = np.argmax(acc)
    vid = np.argmax(val_acc)
    if printFlag:
        print(f'#Epoch {lid}: MIN LOSS  Loss {loss[lid]:.6f}, Train Acc. {acc[lid]:.6f}, Val Acc. {val_acc[lid]:.6f}')
        print(f'#Epoch {aid}: MAX tACC  Loss {loss[aid]:.6f}, Train Acc. {acc[aid]:.6f}, Val Acc. {val_acc[aid]:.6f}')
        print(f'#Epoch {vid}: MAX vACC  Loss {loss[vid]:.6f}, Train Acc. {acc[vid]:.6f}, Val Acc. {val_acc[vid]:.6f}')
    
def plotpkl(path, modelname="Naïve Model", printFlag=True, savepath=None):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    plot(data, modelname, printFlag, savepath)
    
def scatter(df, xaxis='nbar', yaxis='score', color=None):
    """Show the scatter plot of a dataframe.
    :df: dataframe
    :xaxis: column name of the x axis
    :yaxis: column name of the y axis
    :color: column name of the color category (default None)
    """
    data = df
    if color is not None:
        data[color] = df[color].astype('str')
    fig = px.scatter(data, x=xaxis, y=yaxis, color=color, marginal_y="box")
    fig.show()
    return fig
                
    
def heatmap(record, xaxis='nbar', yaxis='width', fix='bandwidth', fixv=0.7):
    """Show the heatmap of a dataframe with fixed value.
    :record: dataframe
    :xaxis: column name of the x axis
    :yaxis: column name of the y axis
    :fix: column name of the fixed attribute
    :fixv: value to be fixed (allow eps=0.01)
    """
    xx = singledim[xaxis]
    yy = singledim[yaxis]
    eps = 0.01
    df = record[abs(record[fix]-fixv)<eps]
    data = [[df[(abs(df[xaxis]-x)<eps) & (abs(df[yaxis]-y)<eps)]['score'].values[0] for x in xx] for y in yy]
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=xx,
        y=yy,
        colorscale='Emrld', zmax=0.8, zmin=0.2))
    fig.update_layout(title=f'{fix}={fixv}', xaxis_title=xaxis, yaxis_title=yaxis, title_x=0.5)  
    fig.show()
    return fig
