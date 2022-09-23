import numpy as np
from sklearn.metrics import accuracy_score

def Normalization(df):
    df_std = df.copy()
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
    return df_std

def Relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Sigmoid_Derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
    
def Softmax(X):       #We used stable softmax
    # e = np.exp(Score)                 #(7,100)
    # _sum = np.sum(e, axis=0)          #(1,100)
    # return e / _sum                   #(7,100)
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps, axis=0)
    
def Y_one_hot(Y, len_last_layer):
    one_hot = np.zeros((len(Y), len_last_layer))
    one_hot[np.arange(Y.size), Y] = 1
    return one_hot.T

def initialize_params(layers, std):
    params = {}
    Delta = {}
    for i in range(1, len(layers)):
        mu = 0
        params['W' + str(i)] = np.random.normal(mu, std, [layers[i], layers[i-1]])
        params['B' + str(i)] = np.random.normal(mu, std, [layers[i], 1]) * 0
        
        Delta['W' + str(i)] = np.zeros([layers[i], layers[i-1]])
        Delta['B' + str(i)] = np.zeros([layers[i], 1])
    return params, Delta

def forward_propagation(X, params, Gaussian_RBF):
    Number_of_layers = len(params)//2
    values = {}
    for i in range(1, Number_of_layers + 1):
        # S is the Score = W * X + B
        # A is the output of layer, A = f(S)
        if i==1:                                                                      # Check is it firt layers or not          
            values['S' + str(i)] = np.dot(params['W' + str(i)], X) + params['B' + str(i)]
            if Gaussian_RBF:
                values['A' + str(i)] = sigmoid(values['S' + str(i)])
            else:
                values['A' + str(i)] = Relu(values['S' + str(i)])
            
        else:
            values['S' + str(i)] = np.dot(params['W' + str(i)], values['A' + str(i-1)]) + params['B' + str(i)]
            if i == Number_of_layers:                                                 # Check is it last layers or not   
                if Gaussian_RBF:
                    values['A' + str(i)] = sigmoid(values['S' + str(i)])
                else:
                    values['A' + str(i)] = values['S' + str(i)]
            else:
                if Gaussian_RBF:
                    values['A' + str(i)] = sigmoid(values['S' + str(i)])
                else:
                    values['A' + str(i)] = Relu(values['S' + str(i)])
    return values

def Compute_Loss(values, Y, params, Gaussian_RBF, len_last_layer):
    Number_of_layers = len(params)//2
    if Gaussian_RBF:
        y_hat = values['A' + str(Number_of_layers)]
        y_true_ohe_hot = Y_one_hot(Y, len_last_layer)
        loss = np.apply_along_axis(np.linalg.norm, 0, y_hat - y_true_ohe_hot)
        loss = np.power(loss, 2)
    else:
        Score_last_layer = values['A' + str(Number_of_layers)]                                         #dim = (7,100)
        loss = np.log(np.sum(np.exp(Score_last_layer), axis=0)) - Score_last_layer[Y, range(len(Y))]

    return loss                                                                                        #dim = (100,)

def backward_propagation(params, values, X, Y, len_last_layer, Gaussian_RBF):
    Number_of_layers = len(params)//2
    m = len(Y)                                                                    #Number_of_samples_in_this_batch
    grads = {}
    for i in range(Number_of_layers, 0, -1):
        if i == Number_of_layers:                                                 #Check is it last layer or not
            if Gaussian_RBF:
                y_hat = values['A' + str(Number_of_layers)]
                y_true_ohe_hot = Y_one_hot(Y, len_last_layer)
                Score_last_layer = values['S' + str(Number_of_layers)] 
                dS = np.multiply(2 * (y_hat - y_true_ohe_hot), Sigmoid_Derivative(Score_last_layer))
            else:
                Score_last_layer = values['A' + str(Number_of_layers)]            #(7,100)
                dS = Softmax(Score_last_layer) - Y_one_hot(Y, len_last_layer)     #(7,100) = (7,100) - (7, 100)
        else:
            dA = np.dot(params['W' + str(i+1)].T, dS)
            if Gaussian_RBF:
                dS = np.multiply(dA, Sigmoid_Derivative(values['A' + str(i)]))    #Sigmoid Derivative
            else:
                dS = np.multiply(dA, np.where(values['A' + str(i)]>=0, 1, 0))     #Relu Derivative
                       
        if i == 1:
            grads['W' + str(i)] = 1/m * np.dot(dS, X.T)
            grads['B' + str(i)] = 1/m * np.sum(dS, axis=1, keepdims=True)
        else:
            grads['W' + str(i)] = 1/m * np.dot(dS,values['A' + str(i-1)].T)
            grads['B' + str(i)] = 1/m * np.sum(dS, axis=1, keepdims=True)
    return grads

def update_params(params, grads, learning_rate, alfa_momentum, Delta):
    Number_of_layers = len(params)//2
    params_updated = {}
    for i in range(1, Number_of_layers + 1):
        Delta['W' + str(i)] = Delta['W' + str(i)] * alfa_momentum - learning_rate * grads['W' + str(i)]   
        Delta['B' + str(i)] = Delta['B' + str(i)] * alfa_momentum - learning_rate * grads['B' + str(i)]
        params_updated['W' + str(i)] = params['W' + str(i)] + Delta['W' + str(i)]
        params_updated['B' + str(i)] = params['B' + str(i)] + Delta['B' + str(i)]
    return params_updated, Delta

def predict(_Data, params, Gaussian_RBF):
    Number_of_layers = len(params)//2
    values = forward_propagation(_Data.T, params, Gaussian_RBF)
    predictions = values['A' + str(Number_of_layers)]
    y_pred = np.argmax(predictions, axis=0)
    return y_pred

def model(X_train, Y_train, X_test, Y_test, layers, Number_of_Epochs, learning_rate, std, alfa_momentum, Batch_size, Gaussian_RBF = False):
    params, Delta = initialize_params(layers, std)
    _Loss_test = []
    _Loss_train = []
    _acc_test = []
    _acc_train = []
    for i in range(Number_of_Epochs):
        Each_Epoch_Loss = np.array([])
        for j in range(int(len(X_train) / Batch_size) + 1):
            X = X_train[j * Batch_size: (j+1)*Batch_size]
            Y = Y_train[j * Batch_size: (j+1)*Batch_size]
            values = forward_propagation(X.T, params, Gaussian_RBF)
            Loss_of_one_batch = Compute_Loss(values, Y.T, params, Gaussian_RBF, layers[-1])
            Each_Epoch_Loss = np.append(Each_Epoch_Loss, Loss_of_one_batch)
            grads = backward_propagation(params, values, X.T, Y.T, layers[-1], Gaussian_RBF)
            params, Delta = update_params(params, grads, learning_rate, alfa_momentum, Delta)
        
        _Loss_train.append(np.mean(Each_Epoch_Loss))
        values = forward_propagation(X_test.T, params, Gaussian_RBF)
        _Loss_test.append(np.mean(Compute_Loss(values, Y_test.T, params, Gaussian_RBF, layers[-1])))
        y_pred_train = predict(X_train, params, Gaussian_RBF)
        y_pred_test = predict(X_test, params, Gaussian_RBF) 
        _acc_train.append(accuracy_score(Y_train, y_pred_train))
        _acc_test.append(accuracy_score(Y_test, y_pred_test))     
    return params, _Loss_train, _Loss_test, _acc_test, _acc_train, y_pred_test
