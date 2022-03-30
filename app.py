"""     Zadanie:
    f(x)" = -11*f(x)
    f(-5) = 0
    f(5) = 15
    wolfram alpha rozwiązanie
    y(x)≈-25.3283*sin(3.31662*(x + 5))
    """

# pydata stack
import numpy as np

import plotly.graph_objects as go

# dash libraries
import dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html

app = dash.Dash(__name__)
server = app.server
app.title = "Metody polowe"
print("runs")

app.layout = html.Div(
    children=[
        html.H1(children="Metody polowe 1"),
        html.P("""Numeryczne rozwiązanie równania: f"(x)=-kf(x)"""),
        html.Div(
            children=[
                "k=",
                dcc.Input(
                    id="k",
                    type="number",
                    placeholder="k",
                    min=1,
                    max=100,
                    step=1,
                    value=11,
                ),
                ",   Dirichlet...y0=",
                dcc.Input(
                    id="Dirichlet",
                    type="number",
                    placeholder="f(First)",
                    min=-100,
                    max=100,
                    step=1,
                    value=0,
                ),
                ",       Neumann...y'n=",
                dcc.Input(
                    id="Neuman",
                    type="number",
                    placeholder="f'(Last)",
                    min=-100,
                    max=100,
                    step=1,
                    value=15,
                ),
                html.P(),
                "f0=",
                dcc.Input(
                    id="Start",
                    type="number",
                    placeholder="First",
                    min=-100,
                    max=100,
                    step=1,
                    value=-5,
                ),
                ",      fn=",
                dcc.Input(
                    id="Stop",
                    type="number",
                    placeholder="Last",
                    min=-100,
                    max=100,
                    step=1,
                    value=5,
                ),
            ]
        ),
        html.Div(
            children=[
                dcc.Graph(id="the_graph"),
                dcc.Slider(
                    id="slider-n",
                    min=0.4,
                    max=3.5,
                    step=0.01,
                    marks={i: "{}".format(round(10**i)) for i in range(4)},
                    value=1.5,
                    updatemode="drag",
                ),
                html.Div(children=[], id="update-output-container"),
                html.Div(children=[], id="error-output-container"),
                dcc.Graph(id="error_graph"),
            ]
        ),
    ]
)


@app.callback(
    Output("the_graph", "figure"),
    Output("update-output-container", "children"),
    Output("error-output-container", "children"),
    Output("error_graph", "figure"),
    Input("slider-n", "value"),
    Input("k", "value"),
    Input("Start", "value"),
    Input("Stop", "value"),
    Input("Dirichlet", "value"),
    Input("Neuman", "value"),
)
def display_value(value, k, start, stop, war_dirichlet, war_neuman):
    m = 10**value
    n = int(m)
    h = (stop - start) / (n - 1)
    arr = np.zeros((n - 2, n))
    for i in range(n - 2):
        arr[i][i] = 1
        arr[i][i + 1] = k * h * h - 2
        arr[i][i + 2] = 1
    dirichlet = np.zeros(n)
    dirichlet[0] = 1
    neuman = np.zeros(n)
    neuman[-1] = 1 / h
    neuman[-2] = -1 / h
    arr = np.vstack([dirichlet, arr])
    arr = np.vstack([arr, neuman])

    b = np.zeros(n)
    b[0] = war_dirichlet
    b[-1] = war_neuman
    B = np.array([b])

    solution = np.linalg.solve(arr, B.T)

    x_axis = np.linspace(start, stop, n)
    y_axis = np.transpose(solution)[0]
    x_solution = np.linspace(start, stop, 100)
    y_solution = -25.3283 * np.sin(3.31662 * (x_solution + 5))
    print("runs")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x_axis, y=y_axis, mode="markers+lines", name="calculated")
    )
    fig.add_trace(
        go.Scatter(
            x=x_solution,
            y=y_solution,
            mode="lines",
            name="""y''=-11y,  y(-5)=0,  y(5)=15""",
        )
    )
    fig.update_layout(
        title="_", xaxis_title="X", yaxis_title="Y", legend_title="Legend"
    )

    def reject_outliers(data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    y_real = -25.3283 * np.sin(3.31662 * (x_axis + 5))
    err = 100 * np.abs((y_axis - y_real) / np.where(y_real == 0, 0.00001, y_real))
    err = reject_outliers(err)

    avg_err = np.nanmean(err)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=x_axis,
            y=err,
            mode="markers+lines",
            name="Blad",
        )
    )
    fig2.update_layout(
        title="Bląd",
        xaxis_title="X",
        yaxis_title="błąd bezwzględny [%]",
        legend_title="Legend Title",
    )

    return (
        fig,
        "Liczba punktów: {}".format(n),
        "Sredni błąd bezwzględny: {} %".format((round(avg_err, 1))),
        fig2,
    )


if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8200, debug=True)