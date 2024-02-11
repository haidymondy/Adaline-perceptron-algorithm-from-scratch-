import tkinter as tk
from tkinter import ttk
from tkinter.ttk import Combobox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# Load the dataset from an Excel file
data = pd.read_excel('Dry_Bean_Dataset.xlsx')

data['MinorAxisLength'].fillna(data['MinorAxisLength'].mean(), inplace=True)

# Define a function for calculating confusion matrix
def calculate_confusion_matrix(actual, predicted):
    tp = np.sum((actual == 1) & (predicted == 1))
    tn = np.sum((actual == -1) & (predicted == -1))
    fp = np.sum((actual == -1) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == -1))
    return tp, tn, fp, fn





# Define the Perceptron model
class Perceptron:
    def __init__(self, select_eta, select_epoch, select_mse_thre, select_bias):
        self.select_eta = select_eta
        self.select_epoch = select_epoch
        self.select_mse_thre = select_mse_thre
        self.select_bias = select_bias


    def calculate_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def calculate_accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def signum(self, x):
        return np.where(x >= 0, 1, -1)

    def fit(self, X, y):
        # Initialize the weights
        if self.select_bias:
            X = np.c_[np.ones(X.shape[0]), X]
        else:
            X = np.c_[np.zeros(X.shape[0]), X]

        self.weights = np.random.rand(X.shape[1])
        mse_list = []  # to store MSE values during training

        for i in range(self.select_epoch):
            net_value = np.dot(X, self.weights)
            Pred_output=self.signum(net_value)
            error = y - Pred_output
            mse = self.calculate_mse(y, Pred_output)
            mse_list.append(mse)

            self.weights += self.select_eta * np.dot(X.T, error)

            if mse < self.select_mse_thre:
                break

        return mse_list

    def predict(self, X):
        if self.select_bias:
            X = np.c_[np.ones(X.shape[0]), X]
        else:
            X = np.c_[np.zeros(X.shape[0]), X]

        net_value = np.dot(X, self.weights)
        predicted_output = self.signum(net_value)

        return predicted_output





# Define the Adaline model
class Adaline:
    def __init__(self, select_eta, select_epoch, select_mse_thre, select_bias):
        self.select_eta = select_eta
        self.select_epoch = select_epoch
        self.select_mse_thre = select_mse_thre
        self.select_bias = select_bias

    def act_linear(self, x):
        return x

    def calculate_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def calculate_accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def fit(self, X, y):
        # Initialize the weights and Add_bias
        if self.select_bias:
            X = np.c_[np.ones(X.shape[0]), X]
        else:
            X = np.c_[np.zeros(X.shape[0]), X]

        self.weights = np.random.rand(X.shape[1])
        mse_list = []

        for i in range(self.select_epoch):
            net_value = np.dot(X, self.weights)
            predic_output = self.act_linear(net_value)
            predic_output = np.where(predic_output >= 0, 1, -1)
            error = y - predic_output

            mse = self.calculate_mse(y, predic_output)
            mse_list.append(mse)

            self.weights += self.select_eta * np.dot(X.T, error)

            # if MSE threshold found
            if mse < self.select_mse_thre:
                break

        return mse_list

    def predict(self, X):
        # Add bias term to the features
        if self.select_bias:
            X = np.c_[np.ones(X.shape[0]), X]
        else:
            X = np.c_[np.zeros(X.shape[0]), X]

        # Calculate the net value
        net_value = np.dot(X, self.weights)
        # Calculate the actual output
        predic_output = self.act_linear(net_value)
        predic_output = np.where(predic_output >= 0, 1, -1)
        return predic_output


def train_and_predict():
    pr = tk.Tk()
    pr.geometry('300x300+700+300')
    pr.resizable(False, False)
    pr.configure(bg='#004a87')
    pr.minsize(10, 10)


    # Retrieve user selections from the GUI components
    feature1 = com1.get()
    feature2 = com2.get()
    class1 = com3.get()
    class2 = com4.get()
    learning_rate = float(en5.get())
    num_epochs = int(en6.get())
    mse_threshold = float(en7.get())
    algorithm_choice = v.get()
    add_bias = g.get()
    #read_bias = float(en8.get())


    # Filter the data for selected classes
    selected_classes = [class1, class2]
    filtered_data = data[data['Class'].isin(selected_classes)]

    # Split the data into features and labels
    X = filtered_data[[feature1, feature2]]
    y = filtered_data['Class']

    # Normalize the features
    X = (X - X.min()) / (X.max() - X.min())

    class_mapping = {class1: 1, class2: -1}
    y = y.map(class_mapping)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




    if algorithm_choice == 1:
        if add_bias == 1:
            # Train the Perceptron model
            perceptron_model = Perceptron(learning_rate, num_epochs, mse_threshold, True)
            mse_list = perceptron_model.fit(X_train, y_train)
            y_pred = perceptron_model.predict(X_test)

            final_mse_perceptron = mse_list[-1]
            print(f'perceptron - Final Mean Squared Error for train (MSE): {final_mse_perceptron}')

            accuracy_perceptron = perceptron_model.calculate_accuracy(y_test, y_pred)
            print(f'perceptron - Accuracy test: {accuracy_perceptron * 100:.2f}%')
            # sample features
            sample_features_perceotron = np.array([[float(en9.get()), float(en10.get())]])
            # scaler = StandardScaler()
            # X = scaler.fit_transform(X)
            sample_features_perceotron = (sample_features_perceotron - sample_features_perceotron.min()) / (sample_features_perceotron.max() - sample_features_perceotron.min())
            # Classify using Perceptron
            class_ids_perceptron = perceptron_model.predict(sample_features_perceotron)

            print("Net Value for perceptron :",
                  np.dot(np.c_[np.ones(sample_features_perceotron.shape[0]), sample_features_perceotron],
                         perceptron_model.weights))

            print("Actual Output for perceptron:", perceptron_model.signum(
                np.dot(np.c_[np.ones(sample_features_perceotron.shape[0]), sample_features_perceotron],
                       perceptron_model.weights)))

            print("perceptron Class IDs:", class_ids_perceptron)

            model = perceptron_model

            model_name = "Perceptron"
        elif add_bias == 0 :
            perceptron_model = Perceptron(learning_rate, num_epochs, mse_threshold,False)
            mse_list = perceptron_model.fit(X_train, y_train)
            y_pred = perceptron_model.predict(X_test)

            final_mse_perceptron = mse_list[-1]
            print(f'perceptron - Final Mean Squared Error for train (MSE): {final_mse_perceptron}')

            accuracy_perceptron = perceptron_model.calculate_accuracy(y_test, y_pred)
            print(f'perceptron - Accuracy test: {accuracy_perceptron * 100:.2f}%')

            # sample features
            sample_features_perceotron = np.array([[float(en9.get()), float(en10.get())]])

            # Classify using Perceptron
            class_ids_perceptron = perceptron_model.predict(sample_features_perceotron)

            print("Net Value for perceptron :",
                  np.dot(np.c_[np.ones(sample_features_perceotron.shape[0]), sample_features_perceotron],
                         perceptron_model.weights))

            print("Actual Output for perceptron:", perceptron_model.signum(
                np.dot(np.c_[np.ones(sample_features_perceotron.shape[0]), sample_features_perceotron],
                       perceptron_model.weights)))

            print("perceptron Class IDs:", class_ids_perceptron)

            model = perceptron_model

            model_name = "Perceptron"


        lbl = tk.Label(pr, text="Perceptron Test Accuracy ", fg='black',
                    bg='#004a87', font=('Helvetic', 15, 'bold'))
        lbl.place(x=20, y=60)
        lbl1 = tk.Label(pr, text={accuracy_perceptron * 100}, fg='white',
                     bg='#004a87', font=('Helvetic', 13, 'bold'))
        lbl1.place(x=110, y=90)


        lbl = tk.Label(pr, text="Perceptron MSE of Train ", fg='black',
                       bg='#004a87', font=('Helvetic', 15, 'bold'))
        lbl.place(x=70, y=120)
        lbl1 = tk.Label(pr, text=final_mse_perceptron, fg='white',
                        bg='#004a87', font=('Helvetic', 13, 'bold'))
        lbl1.place(x=120, y=150)

        lbl = tk.Label(pr, text="Class ID ", fg='black',
                       bg='#004a87', font=('Helvetic', 15, 'bold'))
        lbl.place(x=110, y=180)
        lbl1 = tk.Label(pr, text=class_ids_perceptron, fg='white',
                        bg='#004a87', font=('Helvetic', 13, 'bold'))
        lbl1.place(x=130, y=210)
            ########################################################################################################
    elif algorithm_choice == 2:
        # Train the Adaline model
        if add_bias == 1 :
            adaline_model = Adaline(learning_rate, num_epochs, mse_threshold, True)
            mse_list = adaline_model.fit(X_train, y_train)
            y_pred = adaline_model.predict(X_test)

            final_mse_adalin = mse_list[-1]
            print(f'Adaline - Final Mean Squared Error for train (MSE): {final_mse_adalin}')

            accuracy_adaline = accuracy_score(y_test, y_pred)
            print(f'Adaline - Accuracy test: {accuracy_adaline * 100:.2f}%')

            sample_features_adalin = np.array([[float(en9.get()), float(en10.get())]])
            # Classify using Adaline
            class_ids_adaline = adaline_model.predict(sample_features_adalin)

            print("Net Value for adalin :",np.dot(np.c_[np.ones(sample_features_adalin.shape[0]), sample_features_adalin], adaline_model.weights))
            print("Actual Output for adalin:", adaline_model.act_linear(np.dot(np.c_[np.ones(sample_features_adalin.shape[0]), sample_features_adalin], adaline_model.weights)))
            print("adalin Class IDs:", class_ids_adaline)

            model = adaline_model
            model_name = "Adaline"

        elif add_bias  == 0:
            adaline_model = Adaline(learning_rate, num_epochs, mse_threshold, False)
            mse_list = adaline_model.fit(X_train, y_train)
            y_pred = adaline_model.predict(X_test)

            final_mse_adalin = mse_list[-1]
            print(f'Adaline - Final Mean Squared Error for train (MSE): {final_mse_adalin}')

            accuracy_adaline = accuracy_score(y_test, y_pred)
            print(f'Adaline - Accuracy test: {accuracy_adaline * 100:.2f}%')

            sample_features_adalin = np.array([[float(en9.get()), float(en10.get())]])
            # Classify using Adaline
            class_ids_adaline = adaline_model.predict(sample_features_adalin)

            print("Net Value for adalin :",
                  np.dot(np.c_[np.ones(sample_features_adalin.shape[0]), sample_features_adalin],
                         adaline_model.weights))
            print("Actual Output for adalin:", adaline_model.act_linear(
                np.dot(np.c_[np.ones(sample_features_adalin.shape[0]), sample_features_adalin], adaline_model.weights)))
            print("adalin Class IDs:", class_ids_adaline)

            model = adaline_model
            model_name = "Adaline"

        lbl = tk.Label(pr, text="Adaline Test Accuracy ", fg='black',
                       bg='#004a87', font=('Helvetic', 15, 'bold'))
        lbl.place(x=20, y=60)
        lbl1 = tk.Label(pr, text={accuracy_adaline * 100}, fg='white',
                        bg='#004a87', font=('Helvetic', 13, 'bold'))
        lbl1.place(x=110, y=90)

        lbl = tk.Label(pr, text="Adaline MSE of Train", fg='black',
                       bg='#004a87', font=('Helvetic', 15, 'bold'))
        lbl.place(x=70, y=120)
        lbl1 = tk.Label(pr, text=final_mse_adalin, fg='white',
                        bg='#004a87', font=('Helvetic', 13, 'bold'))
        lbl1.place(x=120, y=150)

        lbl = tk.Label(pr, text="Class ID ", fg='black',
                       bg='#004a87', font=('Helvetic', 15, 'bold'))
        lbl.place(x=110, y=180)
        lbl1 = tk.Label(pr, text=class_ids_adaline, fg='white',
                        bg='#004a87', font=('Helvetic', 13, 'bold'))
        lbl1.place(x=130, y=210)

    # Calculate confusion matrix
    tp, tn, fp, fn = calculate_confusion_matrix(y_test, y_pred)

    # Update or display the results on the GUI
    result_label.config(text=f"{model_name} - Confusion Matrix:\n"
                             f"True Positives (TP): {tp}\n"
                             f"True Negatives (TN): {tn}\n"
                             f"False Positives (FP): {fp}\n"
                             f"False Negatives (FN): {fn}")




    # Define a function for plotting decision boundaries
    def plot_decision_boundary(model, title):
        plt.scatter(X_test[feature1], X_test[feature2], c=y_test, cmap='viridis', marker='x', label='Testing Data')

        w1, w2 = model.weights[1:]
        b = model.weights[0]
        x_line = np.linspace(np.min(X_train[feature2]), np.max(X_train[feature2]), num=100)
        y_line = -(w1 * x_line + b) / w2

        plt.plot(x_line, y_line, color='red')

        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(title)
        plt.legend()
        return plt.show()

    plot_decision_boundary(model, f'{model_name} Classifier')









pro = tk.Tk()
pro.geometry('1000x750+300+1')
pro.resizable(False, False)
pro.title('Service cancellation predictor')
pro.config(background='white')
pro.minsize(500, 500)

fr = tk.Frame(width='1250', height='100', bg='Teal')  # grey  Teal
fr.pack()

v = tk.IntVar()
g = tk.IntVar()
com1 = Combobox(pro, values=['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes'])
com1.place(x=260, y=150)
lbl1 = tk.Label(pro, text='First Feature', fg='black', bg='#c3ecf3', font=(None, 13, 'bold'))
lbl1.place(x=120, y=150)

com2 = Combobox(pro, values=['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes'])
com2.place(x=660, y=150)
lbl2 = tk.Label(pro, text='Second Feature', fg='black', bg='#c3ecf3', font=(None, 13, 'bold'))
lbl2.place(x=500, y=150)

com3 = Combobox(pro, values=['BOMBAY', 'CALI', 'SIRA'])
com3.place(x=260, y=260)
lbl3 = tk.Label(pro, text='First Class', fg='black', bg='#c3ecf3', font=(None, 13, 'bold'))
lbl3.place(x=120, y=260)

com4 = Combobox(pro, values=['BOMBAY', 'CALI', 'SIRA'])
com4.place(x=660, y=260)
lbl4 = tk.Label(pro, text='Second Class', fg='black', bg='#c3ecf3', font=(None, 13, 'bold'))
lbl4.place(x=500, y=260)

en5 = tk.Entry(pro, relief="flat", highlightthickness=1, highlightbackground="gray", highlightcolor="cyan", bg='white')
en5.place(x=260, y=370)
lbl5 = tk.Label(pro, text='Learning Rate', fg='black', bg='#c3ecf3', font=(None, 13, 'bold'))
lbl5.place(x=120, y=370)

en6 = tk.Entry(pro, relief="flat", highlightthickness=1, highlightbackground="gray", highlightcolor="cyan", bg='white')
en6.place(x=660, y=370)
lbl6 = tk.Label(pro, text='Number of epochs', fg='black', bg='#c3ecf3', font=(None, 13, 'bold'))
lbl6.place(x=500, y=370)

en7 = tk.Entry(pro, relief="flat", highlightthickness=1, highlightbackground="gray", highlightcolor="cyan", bg='white')
en7.place(x=260, y=480)
lbl7 = tk.Label(pro, text='MSE threshold', fg='black', bg='#c3ecf3', font=(None, 13, 'bold'))
lbl7.place(x=120, y=480)

lbl9 = tk.Label(pro, text='sample feature1', fg='black', bg='#c3ecf3', font=(None, 13, 'bold'))
lbl9.place(x=500, y=440)
en9 = tk.Entry(pro, relief="flat", highlightthickness=1, highlightbackground="gray", highlightcolor="cyan", bg='white')
en9.place(x=640, y=440)

lbl10 = tk.Label(pro, text='sample feature2', fg='black', bg='#c3ecf3', font=(None, 13, 'bold'))
lbl10.place(x=500, y=480)
en10 = tk.Entry(pro, relief="flat", highlightthickness=1, highlightbackground="gray", highlightcolor="cyan", bg='white')
en10.place(x=640, y=480)



lbl8 = tk.Label(pro, text='Algorithm', fg='black', bg='#c3ecf3', font=(None, 13, 'bold'))
lbl8.place(x=120, y=590)

r1 = ttk.Radiobutton(pro, text='Perceptron', value=1, variable=v)
r1.place(x=230, y=590)

r2 = ttk.Radiobutton(pro, text='Adaline', value=2, variable=v)
r2.place(x=330, y=590)

c = tk.Checkbutton(pro, text='Add Bias', variable=g, onvalue=1, offvalue=0, font=(None, 13, 'bold'), padx=30,
                   background="#c3ecf3")
c.place(x=500, y=600)
# en8 = tk.Entry(pro, relief="flat", highlightthickness=1, highlightbackground="gray", highlightcolor="cyan", bg='white')
# en8.place(x=670, y=600)


bt1 = tk.Button(text='Done', fg='black', bg='#c3ecf3', width='25', height='2',
                font=('Helvetica', 12, 'italic', 'bold'), activebackground='black', activeforeground='white',
                command=train_and_predict)
bt1.place(x=350, y=700)

result_label = tk.Label(pro, text='', fg='black', bg='#c3ecf3', font=('Helvetica', 12))
result_label.place(x=100, y=750)

pro.mainloop()
