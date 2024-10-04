import numpy as np
import plotly.graph_objs as go
import plotly.subplots as sp
from scipy.io import loadmat
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, threshold=0):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
        self.threshold = threshold

    def predict_single(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return 1 if summation > self.threshold else -1

    def predict(self, inputs):
        return np.array([self.predict_single(x) for x in inputs])

    def train(self, training_data, labels, epochs=20):
        errors = []
        for epoch in range(epochs):
            error = 0
            for inputs, label in zip(training_data, labels):
                prediction = self.predict_single(inputs)
                if prediction != label:
                    error += 1
                    self.weights += self.learning_rate * label * inputs
                    self.bias += self.learning_rate * label
            errors.append(error)
        return errors

def load_data(file_path):
    data = loadmat(file_path)
    x = data['X']
    y = data['Y'].ravel()
    y[y == 0] = -1
    return x, y

def split_data(x, y, test_size=0.2):
    np.random.seed(42)
    mask = np.random.rand(len(y)) < (1 - test_size)
    x_train, x_test = x[mask], x[~mask]
    y_train, y_test = y[mask], y[~mask]
    return x_train, y_train, x_test, y_test

def create_plots(perceptron, x_test, y_test, errors, learning_rate, threshold):
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=('Training Error vs. Epochs', 'Decision Boundary'))

    # Plot training error
    fig.add_trace(go.Scatter(x=list(range(1, len(errors) + 1)), y=errors, mode='lines'), row=1, col=1)
    fig.update_xaxes(title_text='Epochs', row=1, col=1)
    fig.update_yaxes(title_text='Training Error', row=1, col=1)

    # Plot decision boundary
    x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
    y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='RdBu', opacity=0.5), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_test[y_test == 1][:, 0], y=x_test[y_test == 1][:, 1], mode='markers', marker=dict(color='blue', symbol='circle'), name='Class 1'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_test[y_test == -1][:, 0], y=x_test[y_test == -1][:, 1], mode='markers', marker=dict(color='red', symbol='circle'), name='Class -1'), row=1, col=2)

    fig.update_xaxes(title_text='Feature 1', row=1, col=2)
    fig.update_yaxes(title_text='Feature 2', row=1, col=2)

    fig.update_layout(height=500, width=1000, title_text=f"Perceptron Results (LR: {learning_rate}, Threshold: {threshold:.2f})")
    return fig

# Dash App Init()
app = dash.Dash(__name__)

# Layout defition
app.layout = html.Div([
    html.H1("Perceptron Dashboard for EE 456 Mini-Project"),
    
    dcc.Tabs([
        dcc.Tab(label='Case 1: Varying Learning Rates', children=[
            html.Div([
                html.H3("Dataset Selection"),
                dcc.RadioItems(
                    id='dataset-selector-1',
                    options=[
                        {'label': 'Linearly Separable', 'value': 'dataset1.mat'},
                        {'label': 'Non-linearly Separable', 'value': 'dataset2.mat'}
                    ],
                    value='dataset1.mat'
                ),
                html.H3("Learning Rate"),
                dcc.Dropdown(
                    id='learning-rate-selector-1',
                    options=[{'label': str(lr), 'value': lr} for lr in [0.01, 0.1, 0.5]],
                    value=0.01,
                    style={'width': '200px'}
                ),
                html.Button('Run Perceptron', id='run-button-1', n_clicks=0),
                dcc.Graph(id='perceptron-plots-1'),
                html.Div(id='accuracy-output-1')
            ]),
        ]),
        dcc.Tab(label='Case 2: Varying Thresholds', children=[
            html.Div([
                html.H3("Dataset Selection"),
                dcc.RadioItems(
                    id='dataset-selector-2',
                    options=[
                        {'label': 'Linearly Separable', 'value': 'dataset1.mat'},
                        {'label': 'Non-linearly Separable', 'value': 'dataset2.mat'}
                    ],
                    value='dataset1.mat'
                ),
                html.H3("Threshold"),
                dcc.Input(id='threshold-input-2', type='number', placeholder='Enter threshold', value=0, step=0.1),
                html.Button('Run Perceptron', id='run-button-2', n_clicks=0),
                dcc.Graph(id='perceptron-plots-2'),
                html.Div(id='accuracy-output-2')
            ]),
        ])
    ])
])

# Callback for Case 1
@app.callback(
    [Output('perceptron-plots-1', 'figure'),
     Output('accuracy-output-1', 'children')],
    [Input('run-button-1', 'n_clicks')],
    [State('dataset-selector-1', 'value'),
     State('learning-rate-selector-1', 'value')]
)
def update_case1(n_clicks, dataset, learning_rate):
    if n_clicks == 0:
        return dash.no_update, dash.no_update

    x, y = load_data(dataset)
    x_train, y_train, x_test, y_test = split_data(x, y)
    
    perceptron = Perceptron(input_size=x_train.shape[1], learning_rate=learning_rate)
    errors = perceptron.train(x_train, y_train)
    
    test_predictions = perceptron.predict(x_test)
    accuracy = np.mean(test_predictions == y_test)
    
    fig = create_plots(perceptron, x_test, y_test, errors, learning_rate, perceptron.threshold)
    accuracy_text = f"Test accuracy: {accuracy:.2f}"

    return fig, accuracy_text

# Callback for Case 2
@app.callback(
    [Output('perceptron-plots-2', 'figure'),
     Output('accuracy-output-2', 'children')],
    [Input('run-button-2', 'n_clicks')],
    [State('dataset-selector-2', 'value'),
     State('threshold-input-2', 'value')]
)
def update_case2(n_clicks, dataset, threshold):
    if n_clicks == 0:
        return dash.no_update, dash.no_update

    x, y = load_data(dataset)
    x_train, y_train, x_test, y_test = split_data(x, y)
    
    perceptron = Perceptron(input_size=x_train.shape[1], threshold=threshold)
    errors = perceptron.train(x_train, y_train)
    
    test_predictions = perceptron.predict(x_test)
    accuracy = np.mean(test_predictions == y_test)
    
    fig = create_plots(perceptron, x_test, y_test, errors, perceptron.learning_rate, threshold)
    accuracy_text = f"Test accuracy: {accuracy:.2f}"

    return fig, accuracy_text

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)